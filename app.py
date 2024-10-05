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

    # Batch text tokenization enabled by default
    direct_tokenize = False
    transformers_direct_tokenize = os.getenv("T2V_TRANSFORMERS_DIRECT_TOKENIZE")
    if (
        transformers_direct_tokenize is not None
        and transformers_direct_tokenize == "true"
        or transformers_direct_tokenize == "1"
    ):
        direct_tokenize = True

    model_dir = "./models/model"

    def get_model_directory() -> (str, bool):
        if os.path.exists(f"{model_dir}/model_name"):
            with open(f"{model_dir}/model_name", "r") as f:
                model_name = f.read()
                return model_name, True
        # Default model directory is ./models/model
        return model_dir, False

    def get_onnx_runtime() -> bool:
        if os.path.exists(f"{model_dir}/onnx_runtime"):
            with open(f"{model_dir}/onnx_runtime", "r") as f:
                onnx_runtime = f.read()
                return onnx_runtime == "true"
        return False

    def log_info_about_onnx(onnx_runtime: bool):
        if onnx_runtime:
            onnx_quantization_info = "missing"
            if os.path.exists(f"{model_dir}/onnx_quantization_info"):
                with open(f"{model_dir}/onnx_quantization_info", "r") as f:
                    onnx_quantization_info = f.read()
            logger.info(
                f"Running ONNX vectorizer with quantized model for {onnx_quantization_info}"
            )

    model_name, use_sentence_transformer_vectorizer = get_model_directory()
    onnx_runtime = get_onnx_runtime()
    log_info_about_onnx(onnx_runtime)

    meta_config = Meta(model_dir, model_name, use_sentence_transformer_vectorizer)
    vec = Vectorizer(
        model_dir,
        cuda_support,
        cuda_core,
        cuda_per_process_memory_fraction,
        meta_config.get_model_type(),
        meta_config.get_architecture(),
        direct_tokenize,
        onnx_runtime,
        use_sentence_transformer_vectorizer,
        model_name,
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
