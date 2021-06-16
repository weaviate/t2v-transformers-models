from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from typing import Optional
import math
import torch
import time

# limit transformer batch size to limit parellel inference, otherwise we run
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, pad_token='[PAD]')
        self.model.resize_token_embeddings(len(self.tokenizer)) # some models like GPT do not have a [PAD] token

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
            print("starting sentence tokenization")
            before = time.time()
            sentences = sent_tokenize(' '.join(text.split(),))
            elapsed = time.time() - before
            print("finished sentence tokenization in {:.2f}".format(elapsed))

            num_sentences = len(sentences)
            print("number of sentences: {}".format(num_sentences))
            number_of_batch_vectors = math.ceil(num_sentences / MAX_BATCH_SIZE)
            print("number of batches: {}".format(number_of_batch_vectors))
            batch_vectors = 0
            for i in range(0, number_of_batch_vectors):
                start_index = i * MAX_BATCH_SIZE
                end_index = start_index + MAX_BATCH_SIZE

                before=time.time()
                print("start tokenizing sentences: {}".format(sentences[start_index:end_index]))
                tokens = self.tokenize(sentences[start_index:end_index])
                elapsed=time.time()-before
                print("tokenizing took {}".format(elapsed))
                print(" --number of tokens: {}".format(tokens['input_ids'].size()))
                if self.cuda:
                    tokens.to(self.cuda_core)
                batch_results = self.model(**tokens)
                pool_method = self.pool_method_from_config(config)
                if pool_method == "cls":
                    batch_vectors += batch_results.last_hidden_state[:, 0, :].sum(0)
                    continue
                if pool_method == "masked_mean":
                    batch_vectors += self.pool_sum(batch_results.last_hidden_state, tokens['attention_mask'])
                    continue
                raise Exception("invalid pooling method '{}'".format(pool_method))


            return batch_vectors.detach()/num_sentences

    def pool_method_from_config(self, config: VectorInputConfig):
        if config is None:
            return DEFAULT_POOL_METHOD

        if config.pooling_strategy is None or config.pooling_strategy == "":
            return DEFAULT_POOL_METHOD

        return config.pooling_strategy


