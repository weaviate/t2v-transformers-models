import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any, Literal, List
from logging import getLogger, Logger
from config import get_cache_settings

import nltk
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    DPRContextEncoder,
    DPRQuestionEncoder,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from cachetools import cached

from config import ST_LOCAL_FILES_ONLY


# limit transformer batch size to limit parallel inference, otherwise we run
# into memory problems
MAX_BATCH_SIZE = 25  # TODO: take from config
DEFAULT_POOL_METHOD = "masked_mean"


class VectorInputConfig(BaseModel):
    pooling_strategy: Optional[str] = None
    task_type: Optional[str] = None

    def __hash__(self):
        return hash((self.pooling_strategy, self.task_type))

    def __eq__(self, other):
        if isinstance(other, VectorInputConfig):
            return (
                self.pooling_strategy == other.pooling_strategy
                and self.task_type == other.task_type
            )
        return False


class VectorInput(BaseModel):
    text: str
    config: Optional[VectorInputConfig] = None

    def __hash__(self):
        return hash((self.text, self.config))

    def __eq__(self, other):
        if isinstance(other, VectorInput):
            return self.text == other.text and self.config == other.config
        return False


class Vectorizer:
    executor: ThreadPoolExecutor

    def __init__(
        self,
        model_path: str,
        cuda_support: bool,
        cuda_core: str,
        cuda_per_process_memory_fraction: float,
        model_type: str,
        architecture: str,
        direct_tokenize: bool,
        onnx_runtime: bool,
        use_sentence_transformers_vectorizer: bool,
        use_sentence_transformers_multi_process: bool,
        model_name: str,
        trust_remote_code: bool,
        workers: int | None,
    ):
        self.executor = ThreadPoolExecutor()
        if onnx_runtime:
            self.vectorizer = ONNXVectorizer(model_path, trust_remote_code)
        else:
            if model_type == "t5" or use_sentence_transformers_vectorizer:
                self.vectorizer = SentenceTransformerVectorizer(
                    model_path,
                    model_name,
                    cuda_core,
                    trust_remote_code,
                    use_sentence_transformers_multi_process,
                    workers,
                )
            else:
                self.vectorizer = HuggingFaceVectorizer(
                    model_path,
                    cuda_support,
                    cuda_core,
                    cuda_per_process_memory_fraction,
                    model_type,
                    architecture,
                    direct_tokenize,
                    trust_remote_code,
                )

    async def vectorize(self, text: str, config: VectorInputConfig, worker: int = 0):
        if isinstance(self.vectorizer, SentenceTransformerVectorizer):
            loop = asyncio.get_event_loop()
            f = loop.run_in_executor(
                self.executor, self.vectorizer.vectorize, text, config, worker
            )
            return await asyncio.wrap_future(f)

        return await asyncio.wrap_future(
            self.executor.submit(self.vectorizer.vectorize, text, config)
        )


class SentenceTransformerVectorizer:
    workers: List[SentenceTransformer]
    available_devices: List[str]
    cuda_core: str
    use_sentence_transformers_multi_process: bool
    pool: dict[Literal["input", "output", "processes"], Any]
    logger: Logger

    def __init__(
        self,
        model_path: str,
        model_name: str,
        cuda_core: str,
        trust_remote_code: bool,
        use_sentence_transformers_multi_process: bool,
        workers: int | None,
    ):
        self.logger = getLogger("uvicorn")
        self.cuda_core = cuda_core
        self.use_sentence_transformers_multi_process = (
            use_sentence_transformers_multi_process
        )
        self.available_devices = self.get_devices(
            workers, self.use_sentence_transformers_multi_process
        )
        self.logger.info(
            f"Sentence transformer vectorizer running with model_name={model_name}, cache_folder={model_path} trust_remote_code:{trust_remote_code}"
        )
        self.workers = []
        for device in self.available_devices:
            model = SentenceTransformer(
                model_name,
                cache_folder=model_path,
                device=device,
                trust_remote_code=trust_remote_code,
                local_files_only=ST_LOCAL_FILES_ONLY,
            )
            model.eval()  # make sure we're in inference mode, not training
            self.workers.append(model)

        if self.use_sentence_transformers_multi_process:
            self.pool = self.workers[0].start_multi_process_pool(
                target_devices=self.get_cuda_devices()
            )
            self.logger.info(
                "Sentence transformer vectorizer is set to use all available devices"
            )
            self.logger.info(
                f"Created pool of {len(self.pool['processes'])} available {'CUDA' if torch.cuda.is_available() else 'CPU'} devices"
            )

    def get_cuda_devices(self) -> List[str] | None:
        if self.cuda_core is not None and self.cuda_core != "":
            return self.cuda_core.split(",")

    def get_devices(
        self,
        workers: int | None,
        use_sentence_transformers_multi_process: bool,
    ) -> List[str | None]:
        if (
            not self.use_sentence_transformers_multi_process
            and self.cuda_core is not None
            and self.cuda_core != ""
        ):
            return self.cuda_core.split(",")
        if use_sentence_transformers_multi_process or workers is None or workers < 1:
            return [None]
        return [None] * workers

    @cached(cache=get_cache_settings())
    def vectorize(self, text: str, config: VectorInputConfig, worker: int = 0):
        if self.use_sentence_transformers_multi_process:
            embedding = self.workers[0].encode_multi_process(
                [text], pool=self.pool, normalize_embeddings=True
            )
            return embedding[0]

        embedding = self.workers[worker].encode(
            [text],
            device=self.available_devices[worker],
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]


class ONNXVectorizer:
    model: ORTModelForFeatureExtraction
    tokenizer: AutoTokenizer

    def __init__(self, model_path, trust_remote_code: bool) -> None:
        onnx_path = Path(model_path)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            file_name="model_quantized.onnx",
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            onnx_path, trust_remote_code=trust_remote_code
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def vectorize(self, text: str, config: VectorInputConfig):
        encoded_input = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        )
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0]


class HuggingFaceVectorizer:
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str
    model_type: str
    direct_tokenize: bool
    trust_remote_code: bool

    def __init__(
        self,
        model_path: str,
        cuda_support: bool,
        cuda_core: str,
        cuda_per_process_memory_fraction: float,
        model_type: str,
        architecture: str,
        direct_tokenize: bool,
        trust_remote_code: bool,
    ):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.cuda_per_process_memory_fraction = cuda_per_process_memory_fraction
        self.model_type = model_type
        self.direct_tokenize = direct_tokenize
        self.trust_remote_code = trust_remote_code

        self.model_delegate: HFModel = ModelFactory.model(
            model_type, architecture, cuda_support, cuda_core, trust_remote_code
        )
        self.model = self.model_delegate.create_model(model_path)

        if self.cuda:
            self.model.to(self.cuda_core)
            if self.cuda_per_process_memory_fraction:
                torch.cuda.set_per_process_memory_fraction(
                    self.cuda_per_process_memory_fraction
                )
        self.model.eval()  # make sure we're in inference mode, not training

        self.tokenizer = self.model_delegate.create_tokenizer(model_path)

        nltk.data.path.append("./nltk_data")

    def tokenize(self, text: str):
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=500,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def get_embeddings(self, batch_results):
        return self.model_delegate.get_embeddings(batch_results)

    def get_batch_results(self, tokens, text):
        return self.model_delegate.get_batch_results(tokens, text)

    def pool_embedding(self, batch_results, tokens, config):
        return self.model_delegate.pool_embedding(batch_results, tokens, config)

    def vectorize(self, text: str, config: VectorInputConfig):
        with torch.no_grad():
            if self.direct_tokenize:
                # create embeddings without tokenizing text
                tokens = self.tokenize(text)
                if self.cuda:
                    tokens.to(self.cuda_core)
                batch_results = self.get_batch_results(tokens, text)
                batch_sum_vectors = self.pool_embedding(batch_results, tokens, config)
                return batch_sum_vectors.detach()
            else:
                # tokenize text
                sentences = sent_tokenize(
                    " ".join(
                        text.split(),
                    )
                )
                num_sentences = len(sentences)
                number_of_batch_vectors = math.ceil(num_sentences / MAX_BATCH_SIZE)
                batch_sum_vectors = 0
                for i in range(0, number_of_batch_vectors):
                    start_index = i * MAX_BATCH_SIZE
                    end_index = start_index + MAX_BATCH_SIZE

                    tokens = self.tokenize(sentences[start_index:end_index])
                    if self.cuda:
                        tokens.to(self.cuda_core)
                    batch_results = self.get_batch_results(
                        tokens, sentences[start_index:end_index]
                    )
                    batch_sum_vectors += self.pool_embedding(
                        batch_results, tokens, config
                    )
                return batch_sum_vectors.detach() / num_sentences


class HFModel:

    def __init__(self, cuda_support: bool, cuda_core: str, trust_remote_code: bool):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.trust_remote_code = trust_remote_code

    def create_tokenizer(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.tokenizer

    def create_model(self, model_path):
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.model

    def get_embeddings(self, batch_results):
        return batch_results[0]

    def get_batch_results(self, tokens, text):
        return self.model(**tokens)

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        pooling_method = self.pool_method_from_config(config)
        if pooling_method == "cls":
            return self.get_embeddings(batch_results)[:, 0, :].sum(0)
        elif pooling_method == "masked_mean":
            return self.pool_sum(
                self.get_embeddings(batch_results), tokens["attention_mask"]
            )
        else:
            raise Exception(f"invalid pooling method '{pooling_method}'")

    def pool_method_from_config(self, config: VectorInputConfig):
        if config is None:
            return DEFAULT_POOL_METHOD

        if config.pooling_strategy is None or config.pooling_strategy == "":
            return DEFAULT_POOL_METHOD

        return config.pooling_strategy

    def get_sum_embeddings_mask(self, embeddings, input_mask_expanded):
        if self.cuda:
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1).to(
                self.cuda_core
            )
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9).to(
                self.cuda_core
            )
            return sum_embeddings, sum_mask
        else:
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings, sum_mask

    def pool_sum(self, embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        )
        sum_embeddings, sum_mask = self.get_sum_embeddings_mask(
            embeddings, input_mask_expanded
        )
        sentences = sum_embeddings / sum_mask
        return sentences.sum(0)


class DPRModel(HFModel):

    def __init__(
        self,
        architecture: str,
        cuda_support: bool,
        cuda_core: str,
        trust_remote_code: bool,
    ):
        super().__init__(cuda_support, cuda_core, trust_remote_code)
        self.model = None
        self.architecture = architecture
        self.trust_remote_code = trust_remote_code

    def create_model(self, model_path):
        if self.architecture == "DPRQuestionEncoder":
            self.model = DPRQuestionEncoder.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
        else:
            self.model = DPRContextEncoder.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
        return self.model

    def get_batch_results(self, tokens, text):
        return self.model(tokens["input_ids"], tokens["attention_mask"])

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        # no pooling needed for DPR
        return batch_results["pooler_output"][0]


class T5Model(HFModel):

    def __init__(self, cuda_support: bool, cuda_core: str, trust_remote_code: bool):
        super().__init__(cuda_support, cuda_core)
        self.model = None
        self.tokenizer = None
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.trust_remote_code = trust_remote_code

    def create_model(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.model

    def create_tokenizer(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path, trust_remote_code=self.trust_remote_code
        )
        return self.tokenizer

    def get_embeddings(self, batch_results):
        return batch_results["encoder_last_hidden_state"]

    def get_batch_results(self, tokens, text):
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]

        target_encoding = self.tokenizer(
            text, padding="longest", max_length=500, truncation=True
        )
        labels = target_encoding.input_ids
        if self.cuda:
            labels = torch.tensor(labels).to(self.cuda_core)
        else:
            labels = torch.tensor(labels)

        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class ModelFactory:

    @staticmethod
    def model(
        model_type,
        architecture,
        cuda_support: bool,
        cuda_core: str,
        trust_remote_code: bool,
    ):
        if model_type == "t5":
            return T5Model(cuda_support, cuda_core, trust_remote_code)
        elif model_type == "dpr":
            return DPRModel(architecture, cuda_support, cuda_core, trust_remote_code)
        else:
            return HFModel(cuda_support, cuda_core, trust_remote_code)
