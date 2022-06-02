#!/usr/bin/env python3

from transformers import AutoModel, AutoTokenizer, AutoConfig
import nltk
import os
import sys

model_name = os.getenv('MODEL_NAME', None)
force_automodel = os.getenv('FORCE_AUTOMODEL', False)
if not model_name:
    print("Fatal: MODEL_NAME is required")
    print("Please set environment variable MODEL_NAME to a HuggingFace model name, see https://huggingface.co/models")
    sys.exit(1)

if force_automodel:
    print(f"Using AutoModel for {model_name} to instantiate model")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name)
model.save("./models/trans")