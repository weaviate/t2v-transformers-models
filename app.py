from fastapi import FastAPI
from pydantic import BaseModel
from vectorizer import Vectorizer
import os

app = FastAPI()

cuda_env = os.getenv("ENABLE_CUDA")
cuda_support = cuda_env is not None and cuda_env == "true" or cuda_env == "1"

vec = Vectorizer('./models/test', cuda_support)

class VectorInput(BaseModel):
    text: str

@app.post("/vectors/")
def read_item(item: VectorInput):
    vector = vec.vectorize(item.text)
    return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
