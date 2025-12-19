# text_model/bert_encoder.py

from transformers import TFAutoModel
from .config import MODEL_NAME

def load_bert():
    model = TFAutoModel.from_pretrained(MODEL_NAME)
    model.trainable = False   # 🔒 freeze BERT
    return model

def get_text_embeddings(encoded_inputs):
    model = load_bert()
    outputs = model(encoded_inputs)
    return outputs.last_hidden_state[:, 0, :]
