#!/usr/bin/env python3

import os
import sys
import nltk
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
from sentence_transformers import SentenceTransformer


model_dir = './models/model'
model_name = os.getenv('MODEL_NAME', None)
force_automodel = os.getenv('FORCE_AUTOMODEL', False)
if not model_name:
    print("Fatal: MODEL_NAME is required")
    print("Please set environment variable MODEL_NAME to a HuggingFace model name, see https://huggingface.co/models")
    sys.exit(1)

if force_automodel:
    print(f"Using AutoModel for {model_name} to instantiate model")

print(f"Downloading model {model_name} from huggingface model hub")
config = AutoConfig.from_pretrained(model_name)
model_type = config.to_dict()['model_type']

if model_type is not None and model_type == "t5":
    SentenceTransformer(model_name, cache_folder=model_dir)
    with open(f"{model_dir}/model_name", "w") as f:
        f.write(model_name.replace("/", "_"))
else:
    if config.architectures and not force_automodel:
        print(f"Using class {config.architectures[0]} to load model weights")
        mod = __import__('transformers', fromlist=[config.architectures[0]])
        try:
            klass_architecture = getattr(mod, config.architectures[0])
            model = klass_architecture.from_pretrained(model_name)
        except AttributeError:
            print(f"{config.architectures[0]} not found in transformers, fallback to AutoModel")
            model = AutoModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

nltk.download('punkt', download_dir='./nltk_data')
