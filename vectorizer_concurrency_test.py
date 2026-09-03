"""Concurrency regression tests for the shared tokenizer (issue #123).

Requests are served from a shared ThreadPoolExecutor and share one tokenizer.
On the first calls transformers reconfigures the Rust backend, which needs a
mutable borrow and raises "Already borrowed" if another thread is inside
encode_batch at that moment.
"""

import sys
import threading

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import vectorizer as vectorizer_module
from vectorizer import HuggingFaceVectorizer, ONNXVectorizer, VectorInputConfig

THREADS = 12
ATTEMPTS = 30

# Long enough that encode_batch keeps its borrow while the GIL is available to
# the other threads.
LONG_TEXT = " ".join(f"w{i % 500}" for i in range(50_000))


@pytest.fixture(autouse=True)
def frequent_gil_switches():
    # The gap between reading the tokenizer config and rewriting it is a few
    # bytecodes wide; the default 5ms interval almost never switches inside it.
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


def build_tokenizer():
    words = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"] + [f"w{i}" for i in range(500)]
    backend = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=512,
    )


class StubModel:
    def __call__(self, input_ids, attention_mask, **kwargs):
        batch, length = input_ids.shape
        return (torch.ones(batch, length, 8),)

    def to(self, device):
        return self

    def eval(self):
        return self


class StubLoader:
    def __init__(self, factory):
        self.factory = factory

    def from_pretrained(self, *args, **kwargs):
        return self.factory()


class StubDelegate(vectorizer_module.HFModel):
    def create_model(self, model_path):
        self.model = StubModel()
        return self.model

    def create_tokenizer(self, model_path):
        self.tokenizer = build_tokenizer()
        return self.tokenizer


@pytest.fixture
def make_onnx_vectorizer(monkeypatch):
    monkeypatch.setattr(
        vectorizer_module, "ORTModelForFeatureExtraction", StubLoader(StubModel)
    )
    monkeypatch.setattr(vectorizer_module, "AutoTokenizer", StubLoader(build_tokenizer))
    return lambda: ONNXVectorizer("model", trust_remote_code=False)


@pytest.fixture
def make_huggingface_vectorizer(monkeypatch):
    monkeypatch.setattr(
        vectorizer_module.ModelFactory,
        "model",
        staticmethod(lambda *args, **kwargs: StubDelegate(False, "", False)),
    )
    return lambda: HuggingFaceVectorizer(
        model_path="model",
        cuda_support=False,
        cuda_core="",
        cuda_per_process_memory_fraction=0.0,
        model_type="bert",
        architecture="BertModel",
        direct_tokenize=True,
        trust_remote_code=False,
    )


def collect_failures(make_vectorizer, call):
    """Hammer a freshly built vectorizer from many threads, ATTEMPTS times over."""
    failures = []
    for _ in range(ATTEMPTS):
        target = make_vectorizer()
        start = threading.Barrier(THREADS)

        def worker():
            start.wait()
            try:
                call(target)
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker) for _ in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return failures


def test_onnx_vectorize_survives_concurrent_cold_start(make_onnx_vectorizer):
    failures = collect_failures(
        make_onnx_vectorizer,
        lambda target: target.vectorize(LONG_TEXT, VectorInputConfig()),
    )
    assert not failures, f"{len(failures)} calls failed: {sorted(set(failures))}"


def test_huggingface_tokenize_survives_concurrent_cold_start(
    make_huggingface_vectorizer,
):
    failures = collect_failures(
        make_huggingface_vectorizer, lambda target: target.tokenize(LONG_TEXT)
    )
    assert not failures, f"{len(failures)} calls failed: {sorted(set(failures))}"
