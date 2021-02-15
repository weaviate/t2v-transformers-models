#!/usr/bin/env python3

from transformers import AutoModel, AutoTokenizer
import nltk

model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model.save_pretrained('./models/test')
tokenizer.save_pretrained('./models/test')

nltk.download('punkt')
