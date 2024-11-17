import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

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


# limit transformer batch size to limit parallel inference, otherwise we run
# into memory problems
MAX_BATCH_SIZE = 25  # TODO: take from config
DEFAULT_POOL_METHOD = "masked_mean"


class VectorInputConfig(BaseModel):
    pooling_strategy: Optional[str] = None
    task_type: Optional[str] = None


class VectorInput(BaseModel):
    text: str
    config: Optional[VectorInputConfig] = None


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
        use_sentence_transformer_vectorizer: bool,
        model_name: str,
        trust_remote_code: bool,
    ):
        self.executor = ThreadPoolExecutor()
        self.vectorizer = ONNXVectorizer(model_path, trust_remote_code)

    async def vectorize(self, text: str, config: VectorInputConfig):
        return await asyncio.wrap_future(
            self.executor.submit(self.vectorizer.vectorize, text, config)
        )


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
