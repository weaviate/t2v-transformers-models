from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from vectorizer import Vectorizer
import os

app = FastAPI()

cuda_env = os.getenv("ENABLE_CUDA")
cuda_support = cuda_env is not None and cuda_env == "true" or cuda_env == "1"

vec = Vectorizer('./models/test', cuda_support)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT

class VectorInput(BaseModel):
    text: str

@app.post("/vectors/")
def read_item(item: VectorInput, response: Response):
    try:
        vector = vec.vectorize(item.text)
        return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
