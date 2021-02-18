from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
import math
import torch
import time

# limit transformer batch size to limit parellel inference, otherwise we run
# into memory problems
MAX_BATCH_SIZE = 25  # TODO: take from config

class Vectorizer:
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool

    def __init__(self, model_path: str, cuda_support: bool):
        self.cuda = cuda_support
        self.model = AutoModel.from_pretrained(model_path)
        if self.cuda:
            self.model.to('cuda')
        self.model.eval() # make sure we're in inference mode, not training

        self.tokenizer = AutoTokenizer.from_pretrained("./models/test")

    def tokenize(self, text:str):
        return self.tokenizer(text, padding=True, truncation=True, max_length=500, 
                add_special_tokens = True, return_tensors="pt")


    def vectorize(self, text: str):
        with torch.no_grad():
            print("starting sentence tokenization")
            before = time.time()
            sentences = sent_tokenize(' '.join(text.split(),))
            elapsed = time.time() - before
            print("finished sentence tokenization in {:.2f}".format(elapsed))

            VECTOR_DIM = 768  # TODO: make dynamic

            print("number of sentences: {}".format(len(sentences)))
            number_of_batch_vectors = math.ceil(len(sentences) / MAX_BATCH_SIZE)
            print("number of batches: {}".format(number_of_batch_vectors))
            batch_vectors = torch.Tensor(number_of_batch_vectors, VECTOR_DIM)
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
                    tokens.to('cuda')
                batch_results = self.model(**tokens)
                batch_vectors[i] = batch_results[0].mean(0).mean(0).detach()

            return batch_vectors.mean(0)
