# user_model/mlp_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from .config import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM

def build_user_mlp():
    """
    Builds an MLP model for user behavior modeling.
    """
    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(HIDDEN_DIM, activation="relu"),
        layers.Dense(HIDDEN_DIM, activation="relu"),
        layers.Dense(OUTPUT_DIM, activation="relu")
    ])

    return model
