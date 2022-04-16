import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from vectorizer import Vectorizer, VectorInput
from meta import Meta


app = FastAPI()
vec : Vectorizer
meta_config : Meta
logger = getLogger('uvicorn')


@app.on_event("startup")
def startup_event():
    global vec
    global meta_config

    cuda_env = os.getenv("ENABLE_CUDA")
    cuda_support=False
    cuda_core=""

    inferentia_env = os.getenv("ENABLE_INFERENTIA")
    inferentia_support=False

    if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
        cuda_support=True
        cuda_core = os.getenv("CUDA_CORE")
        if cuda_core is None or cuda_core == "":
            cuda_core = "cuda:0"
        logger.info(f"CUDA_CORE set to {cuda_core}")
    elif inferentia_env is not None and inferentia_env == "true" or inferentia_env == "1":
        inferentia_support=True
        logger.info(f"Inferentia support is set")
    else:
        logger.info("Running on CPU")

    meta_config = Meta('./models/model')
    vec = Vectorizer('./models/model', cuda_support, cuda_core, inferentia_support,
                     meta_config.getModelType(), meta_config.get_architecture(), logger)


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
        logger.exception(
            'Something went wrong while vectorizing data.'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
