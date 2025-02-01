import os
from typing import Optional, List
from logging import getLogger
from fastapi import FastAPI, Depends, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Union
from config import (
    TRUST_REMOTE_CODE,
    get_allowed_tokens,
    get_use_sentence_transformers_multi_process,
    get_t2v_transformers_direct_tokenize,
)
from vectorizer import Vectorizer, VectorInput
from meta import Meta
import torch


logger = getLogger("uvicorn")

vec: Vectorizer
meta_config: Meta

get_bearer_token = HTTPBearer(auto_error=False)
allowed_tokens: List[str] = None
current_worker = 0
available_workers = 1


def is_authorized(auth: Optional[HTTPAuthorizationCredentials]) -> bool:
    if allowed_tokens is not None and (
        auth is None or auth.credentials not in allowed_tokens
    ):
        return False
    return True


def get_worker():
    if available_workers == 1:
        return 0
    else:
        global current_worker
        if current_worker >= 1_000_000_000:
            current_worker = 0
        worker = current_worker % available_workers
        current_worker += 1
        return worker


async def lifespan(app: FastAPI):
    global vec
    global meta_config
    global allowed_tokens
    global available_workers

    allowed_tokens = get_allowed_tokens()

    model_dir = "./models/model"

    def get_model_name() -> Union[str, bool]:
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

    def get_trust_remote_code() -> bool:
        if os.path.exists(f"{model_dir}/trust_remote_code"):
            with open(f"{model_dir}/trust_remote_code", "r") as f:
                trust_remote_code = f.read()
                return trust_remote_code == "true"
        return TRUST_REMOTE_CODE

    def log_info_about_onnx(onnx_runtime: bool):
        if onnx_runtime:
            onnx_quantization_info = "missing"
            if os.path.exists(f"{model_dir}/onnx_quantization_info"):
                with open(f"{model_dir}/onnx_quantization_info", "r") as f:
                    onnx_quantization_info = f.read()
            logger.info(
                f"Running ONNX vectorizer with quantized model for {onnx_quantization_info}"
            )

    model_name, use_sentence_transformers_vectorizer = get_model_name()
    onnx_runtime = get_onnx_runtime()
    trust_remote_code = get_trust_remote_code()

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
    # Use all sentence transformers multi process
    use_sentence_transformers_multi_process = (
        get_use_sentence_transformers_multi_process()
    )

    if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
        cuda_support = True
        cuda_core = os.getenv("CUDA_CORE")
        if cuda_core is None or cuda_core == "":
            if (
                use_sentence_transformers_vectorizer
                and use_sentence_transformers_multi_process
                and torch.cuda.is_available()
            ):
                available_workers = torch.cuda.device_count()
                cuda_core = ",".join([f"cuda:{i}" for i in range(available_workers)])
            else:
                cuda_core = "cuda:0"
        logger.info(f"CUDA_CORE set to {cuda_core}")
    else:
        logger.info("Running on CPU")

    # Batch text tokenization enabled by default
    direct_tokenize = get_t2v_transformers_direct_tokenize()

    log_info_about_onnx(onnx_runtime)

    meta_config = Meta(
        model_dir,
        model_name,
        use_sentence_transformers_vectorizer,
        trust_remote_code,
    )
    vec = Vectorizer(
        model_dir,
        cuda_support,
        cuda_core,
        cuda_per_process_memory_fraction,
        meta_config.get_model_type(),
        meta_config.get_architecture(),
        direct_tokenize,
        onnx_runtime,
        use_sentence_transformers_vectorizer,
        use_sentence_transformers_multi_process,
        model_name,
        trust_remote_code,
        available_workers,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta(
    response: Response,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
):
    if is_authorized(auth):
        return meta_config.get()
    else:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Unauthorized"}


@app.post("/vectors")
@app.post("/vectors/")
async def vectorize(
    item: VectorInput,
    response: Response,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
):
    if is_authorized(auth):
        try:
            vector = await vec.vectorize(item.text, item.config, get_worker())
            return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
        except Exception as e:
            logger.exception("Something went wrong while vectorizing data.")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": str(e)}
    else:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Unauthorized"}
