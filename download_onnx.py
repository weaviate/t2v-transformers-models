#!/usr/bin/env python3

import os
import sys
from transformers import (
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from pathlib import Path


model_dir = "./models/model"
nltk_dir = "./nltk_data"
model_name = os.getenv("MODEL_NAME", None)
force_automodel = os.getenv("FORCE_AUTOMODEL", False)
trust_remote_code = os.getenv("TRUST_REMOTE_CODE", False)
if not model_name:
    print("Fatal: MODEL_NAME is required")
    print(
        "Please set environment variable MODEL_NAME to a HuggingFace model name, see https://huggingface.co/models"
    )
    sys.exit(1)

if force_automodel:
    print(f"Using AutoModel for {model_name} to instantiate model")

onnx_runtime = os.getenv("ONNX_RUNTIME")
if not onnx_runtime:
    onnx_runtime = "false"

onnx_cpu_arch = os.getenv("ONNX_CPU")
if not onnx_cpu_arch:
    onnx_cpu_arch = "arm64"

use_sentence_transformers_vectorizer = os.getenv("USE_SENTENCE_TRANSFORMERS_VECTORIZER")
if not use_sentence_transformers_vectorizer:
    use_sentence_transformers_vectorizer = "false"

print(
    f"Downloading MODEL_NAME={model_name} with FORCE_AUTOMODEL={force_automodel} ONNX_RUNTIME={onnx_runtime} ONNX_CPU={onnx_cpu_arch}"
)


def download_onnx_model(
    model_name: str, model_dir: str, trust_remote_code: bool = False
):
    # Download model and tokenizer
    onnx_path = Path(model_dir)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_name, from_transformers=True, trust_remote_code=trust_remote_code
    )
    # Save model
    ort_model.save_pretrained(onnx_path)

    def save_to_file(filepath: str, content: str):
        with open(filepath, "w") as f:
            f.write(content)

    def save_quantization_info(arch: str):
        save_to_file(f"{model_dir}/onnx_quantization_info", arch)

    def quantization_config(onnx_cpu_arch: str):
        if onnx_cpu_arch.lower() == "avx512_vnni":
            print("Quantize Model for x86_64 (amd64) (avx512_vnni)")
            save_quantization_info("AVX-512")
            return AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            )
        if onnx_cpu_arch.lower() == "arm64":
            print(f"Quantize Model for ARM64")
            save_quantization_info("ARM64")
            return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        # default is AMD64 (AVX2)
        print(f"Quantize Model for x86_64 (amd64) (AVX2)")
        save_quantization_info("amd64 (AVX2)")
        return AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    # Quantize the model / convert to ONNX
    qconfig = quantization_config(onnx_cpu_arch)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    # Apply dynamic quantization on the model
    quantizer.quantize(save_dir=onnx_path, quantization_config=qconfig)
    # Remove model.onnx file, leave only model_quantized.onnx
    if os.path.isfile(f"{model_dir}/model.onnx"):
        os.remove(f"{model_dir}/model.onnx")
    # Save information about ONNX runtime
    save_to_file(f"{model_dir}/onnx_runtime", onnx_runtime)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(onnx_path)


if onnx_runtime == "true":
    download_onnx_model(model_name, model_dir, trust_remote_code)