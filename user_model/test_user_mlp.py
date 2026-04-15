# user_model/test_user_mlp.py

import tensorflow as tf
from user_model.mlp_model import build_user_mlp

dummy_users = tf.random.normal((2, 6))

model = build_user_mlp()
user_embeddings = model(dummy_users)

print("User embedding shape:", user_embeddings.shape)
