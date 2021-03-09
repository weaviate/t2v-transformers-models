#!/usr/bin/env python3

from transformers import AutoModel, AutoTokenizer
import nltk
import os
import sys

model_name = os.getenv('MODEL_NAME')
if model_name is None or model_name == "":
    print("Fatal: MODEL_NAME is required")
    sys.exit(1)

print("Downloading model {} from huggingface model hub".format(model_name))

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained('./models/model')
tokenizer.save_pretrained('./models/model')

nltk.download('punkt')
