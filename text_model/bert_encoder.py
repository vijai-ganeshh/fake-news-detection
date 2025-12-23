import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


# Load tokenizer and model ONCE (important for performance)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased")


def get_text_embeddings(texts):
    """
    Convert a list of texts into BERT embeddings (768-dim)
    Returns NumPy array of shape (batch_size, 768)
    """

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="tf"
    )

    outputs = model(**inputs)

    # Use [CLS] token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    return cls_embeddings.numpy()
