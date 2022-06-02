import os
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

class TextsInput(BaseModel):
    text: list[str]

class TextInput(BaseModel):
    text: str

# For reasons, app can not import the raw function, it must be wrapped in a classs
class Transformer:
    def load_model(self, model_path, device_str):
        model = SentenceTransformer(os.getcwd()+model_path, device=device_str) # Library claims to try and use cuda/gpu when availible
        def encode(text):
            return model.encode(text) # batch size paramter seems to have significat impact on performance
        return encode


