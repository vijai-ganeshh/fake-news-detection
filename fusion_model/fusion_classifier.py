# fusion_model/fusion_classifier.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_fusion_model(
    text_dim=768,
    user_dim=64,
    hidden_dim=128
):
    """
    Builds a fusion model that combines text and user embeddings
    and predicts fake / real news.
    """

    # Input layers
    text_input = layers.Input(shape=(text_dim,), name="text_embedding")
    user_input = layers.Input(shape=(user_dim,), name="user_embedding")

    # Combine embeddings
    fused_features = layers.Concatenate()([text_input, user_input])

    # Fusion layers
    x = layers.Dense(hidden_dim, activation="relu")(fused_features)
    x = layers.Dense(hidden_dim // 2, activation="relu")(x)

    # Output layer
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(
        inputs=[text_input, user_input],
        outputs=output
    )

    return model
