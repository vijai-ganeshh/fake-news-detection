# training/train_text_fusion.py

import tensorflow as tf
import pandas as pd
import numpy as np

# -----------------------------
# Project imports
# -----------------------------
from text_model.tokenizer import tokenize_texts
from text_model.bert_encoder import get_text_embeddings
from user_model.mlp_model import build_user_mlp
from fusion_model.fusion_classifier import build_fusion_model

# -----------------------------
# 1. Load LIAR dataset
# -----------------------------
def load_liar_data(path):
    """
    Loads the LIAR dataset TSV file.
    """
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party", "barely_true",
        "false", "half_true", "mostly_true",
        "pants_fire", "context"
    ]
    return df


print("Loading LIAR dataset...")
df = load_liar_data("data/raw/liar/train.tsv")

# -----------------------------
# 2. Convert labels to binary
# -----------------------------
print("Converting labels to binary...")

fake_labels = ["pants-fire", "false", "barely-true"]

df["binary_label"] = df["label"].apply(
    lambda x: 1 if x in fake_labels else 0
)

texts = df["statement"].astype(str).tolist()
labels = df["binary_label"].values

print(f"Total samples: {len(texts)}")

# -----------------------------
# 3. Dummy user features (temporary)
# -----------------------------
print("Generating dummy user features...")

NUM_USER_FEATURES = 6
user_features = tf.random.normal(
    shape=(len(texts), NUM_USER_FEATURES)
)

# -----------------------------
# 4. Tokenize text
# -----------------------------
print("Tokenizing text...")
encoded_texts = tokenize_texts(texts)

# -----------------------------
# 5. Get text embeddings in batches (FIX for OOM)
# -----------------------------
def batch_text_embeddings(encoded_inputs, batch_size=8):
    """
    Generates BERT embeddings in small batches to avoid OOM.
    """
    all_embeddings = []

    num_samples = encoded_inputs["input_ids"].shape[0]

    for i in range(0, num_samples, batch_size):
        batch = {
            k: v[i:i + batch_size]
            for k, v in encoded_inputs.items()
        }

        batch_emb = get_text_embeddings(batch)
        all_embeddings.append(batch_emb)

        print(f"Processed text batch {i} to {min(i + batch_size, num_samples)}")

    return tf.concat(all_embeddings, axis=0)


print("Generating text embeddings (batched)...")
text_embeddings = batch_text_embeddings(encoded_texts, batch_size=8)

# -----------------------------
# 6. User embeddings (MLP)
# -----------------------------
print("Building user model and embeddings...")
user_model = build_user_mlp()
user_embeddings = user_model(user_features)

# -----------------------------
# 7. Fusion model
# -----------------------------
print("Building fusion model...")
fusion_model = build_fusion_model()

fusion_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 8. Train fusion model
# -----------------------------
print("Starting training...")

fusion_model.fit(
    [text_embeddings, user_embeddings],
    labels,
    epochs=3,
    batch_size=16
)

print("Training completed.")
