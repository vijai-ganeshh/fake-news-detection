# text_model/tokenizer.py

from transformers import AutoTokenizer
from .config import MODEL_NAME, MAX_LENGTH

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_texts(texts):
    tokenizer = load_tokenizer()
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )
