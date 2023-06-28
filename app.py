import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from vectorizer import Vectorizer, VectorInput
from meta import Meta


app = FastAPI()
vec: Vectorizer
meta_config: Meta
logger = getLogger("uvicorn")


@app.on_event("startup")
def startup_event():
    global vec
    global meta_config

    cuda_env = os.getenv("ENABLE_CUDA")
    cuda_per_process_memory_fraction = 1.0
    if "CUDA_PER_PROCESS_MEMORY_FRACTION" in os.environ:
        try:
            cuda_per_process_memory_fraction = float(
                os.getenv("CUDA_PER_PROCESS_MEMORY_FRACTION")
            )
        except ValueError:
            logger.error(
                f"Invalid CUDA_PER_PROCESS_MEMORY_FRACTION (should be between 0.0-1.0)"
            )
    if 0.0 <= cuda_per_process_memory_fraction <= 1.0:
        logger.info(
            f"CUDA_PER_PROCESS_MEMORY_FRACTION set to {cuda_per_process_memory_fraction}"
        )
    cuda_support = False
    cuda_core = ""

    if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
        cuda_support = True
        cuda_core = os.getenv("CUDA_CORE")
        if cuda_core is None or cuda_core == "":
            cuda_core = "cuda:0"
        logger.info(f"CUDA_CORE set to {cuda_core}")
    else:
        logger.info("Running on CPU")

    # Split input in sentences by default
    # And perform tokenization in batches
    shall_split_in_sentences = True
    env_shall_split_in_sentences = os.getenv("T2V_SHALL_SPLIT_IN_SENTENCES")
    if (
        env_shall_split_in_sentences is not None
        and env_shall_split_in_sentences == "false"
        or env_shall_split_in_sentences == "0"
    ):
        shall_split_in_sentences = False

    if not shall_split_in_sentences:
        logger.warn(
            f"Configured not to split input into sentences. Inputs will be truncated if they exceed the models context length."
        )

    meta_config = Meta("./models/model")
    vec = Vectorizer(
        "./models/model",
        cuda_support,
        cuda_core,
        cuda_per_process_memory_fraction,
        meta_config.getModelType(),
        meta_config.get_architecture(),
        shall_split_in_sentences,
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
