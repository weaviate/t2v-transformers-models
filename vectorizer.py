from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize

class Vectorizer:
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool

    def __init__(self, model_path: str, cuda_support: bool):
        self.cuda = cuda_support
        self.model = AutoModel.from_pretrained(model_path)
        if self.cuda:
            self.model.to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained("./models/test")

    def tokenize(self, text:str):
        return self.tokenizer(sent_tokenize(text), padding=True, truncation=True, max_length=500, 
                add_special_tokens = True, return_tensors="pt")


    def vectorize(self, text: str):
        tokens = self.tokenize(text)
        if self.cuda:
            tokens.to('cuda')
        outputs = self.model(**tokens)

        return outputs[0].mean(0).mean(0).detach()
