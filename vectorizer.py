from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from typing import Optional
import math
import torch
import time

# limit transformer batch size to limit parallel inference, otherwise we run
# into memory problems
MAX_BATCH_SIZE = 25  # TODO: take from config
DEFAULT_POOL_METHOD="masked_mean"

class VectorInputConfig(BaseModel):
    pooling_strategy: str


class VectorInput(BaseModel):
    text: str
    config: Optional[VectorInputConfig] = None

class Vectorizer:
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str

    def __init__(self, model_path: str, cuda_support: bool, cuda_core: str):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.model = AutoModel.from_pretrained(model_path)
        if self.cuda:
            self.model.to(self.cuda_core)
        self.model.eval() # make sure we're in inference mode, not training

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(self, text:str):
        return self.tokenizer(text, padding=True, truncation=True, max_length=500, 
                add_special_tokens = True, return_tensors="pt")

    def pool_sum(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentences = sum_embeddings / sum_mask
        return sentences.sum(0)

    async def vectorize(self, text: str, config: VectorInputConfig):
        with torch.no_grad():
            sentences = sent_tokenize(' '.join(text.split(),))
            num_sentences = len(sentences)
            number_of_batch_vectors = math.ceil(num_sentences / MAX_BATCH_SIZE)
            batch_sum_vectors = 0
            for i in range(0, number_of_batch_vectors):
                start_index = i * MAX_BATCH_SIZE
                end_index = start_index + MAX_BATCH_SIZE

                tokens = self.tokenize(sentences[start_index:end_index])
                if self.cuda:
                    tokens.to(self.cuda_core)
                batch_results = self.model(**tokens)
                pool_method = self.pool_method_from_config(config)
                if pool_method == "cls":
                    batch_sum_vectors += batch_results[0][:, 0, :].sum(0)
                    continue
                if pool_method == "masked_mean":
                    batch_sum_vectors += self.pool_sum(batch_results[0], tokens['attention_mask'])
                    continue
                raise Exception(f"invalid pooling method '{pool_method}'")

            return batch_sum_vectors.detach() / num_sentences

    def pool_method_from_config(self, config: VectorInputConfig):
        if config is None:
            return DEFAULT_POOL_METHOD

        if config.pooling_strategy is None or config.pooling_strategy == "":
            return DEFAULT_POOL_METHOD

        return config.pooling_strategy


