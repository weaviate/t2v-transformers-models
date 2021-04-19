from fastapi import FastAPI, Response, status
from vectorizer import Vectorizer, VectorInput
from meta import Meta
import os

app = FastAPI()

cuda_env = os.getenv("ENABLE_CUDA")
cuda_support=False
cuda_core=""
if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
    cuda_support=True
    cuda_core = os.getenv("CUDA_CORE")
    if cuda_core is None or cuda_core == "":
        cuda_core = "cuda:0"
    print("[INFO] cuda core set to {}".format(cuda_core))
else:
    print("[INFO] running on CPU")

vec = Vectorizer('./models/model', cuda_support, cuda_core)
meta_config = Meta('./models/model')


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    return meta_config.get()


@app.post("/vectors/")
async def read_item(item: VectorInput, response: Response):
    try:
        vector = await vec.vectorize(item.text, item.config)
        return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
