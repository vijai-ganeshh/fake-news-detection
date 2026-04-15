from text_model.tokenizer import tokenize_texts
from text_model.bert_encoder import get_text_embeddings

texts = ["This news article is completely fake"]

encoded = tokenize_texts(texts)
embeddings = get_text_embeddings(encoded)

print(embeddings.shape)
