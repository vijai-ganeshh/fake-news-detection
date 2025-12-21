import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_text_embeddings(encoded_inputs):
    """
    Returns CLS embeddings from BERT (TensorFlow-native)
    """
    model = TFAutoModel.from_pretrained(
        MODEL_NAME,
        from_pt=False,        # force TensorFlow weights
        use_safetensors=False
    )

    outputs = model(encoded_inputs, training=False)

    # CLS token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    return cls_embeddings
