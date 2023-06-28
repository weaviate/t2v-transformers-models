from logging import getLogger
from fastapi import FastAPI, Response, status
from vectorizer import Vectorizer, VectorInput
from meta import Meta
from config import T2VConfig

app = FastAPI()
vec: Vectorizer
meta_config: Meta
logger = getLogger("uvicorn")


@app.on_event("startup")
def startup_event():
    global vec
    global meta_config

    config = T2VConfig.from_env()

    if config.cuda_config.enable_cuda:
        logger.info("Running on CUDA")
        logger.info(f"CUDA_CORE set to {config.cuda_config.cuda_core}")
        logger.info(
            f"CUDA_PER_PROCESS_MEMORY_FRACTION set to {config.cuda_configcuda_per_process_memory_fraction}"
        )
    else:
        logger.info("Running on CPU")

    if not config.shall_split_in_sentences:
        logger.warn(
            f"Configured not to split input into sentences. Inputs will be truncated if they exceed the models context length."
        )

    meta_config = Meta("./models/model")
    vec = Vectorizer(
        "./models/model",
        config.cuda_config.enable_cuda,
        config.cuda_config.cuda_core,
        config.cuda_config.cuda_per_process_memory_fraction,
        meta_config.getModelType(),
        meta_config.get_architecture(),
        config.shall_split_in_sentences,
    )


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    return meta_config.get()


@app.post("/vectors")
@app.post("/vectors/")
async def read_item(item: VectorInput, response: Response):
    try:
        vector = await vec.vectorize(item.text, item.config)
        return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
    except Exception as e:
        logger.exception("Something went wrong while vectorizing data.")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
