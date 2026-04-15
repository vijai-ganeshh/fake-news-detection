# fusion_model/test_fusion.py

import tensorflow as tf
from fusion_model.fusion_classifier import build_fusion_model

# Dummy embeddings
dummy_text_embeddings = tf.random.normal((2, 768))
dummy_user_embeddings = tf.random.normal((2, 64))

model = build_fusion_model()

predictions = model([dummy_text_embeddings, dummy_user_embeddings])

print("Fusion output shape:", predictions.shape)
