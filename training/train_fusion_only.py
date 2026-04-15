import numpy as np
from fusion_model.fusion_classifier import build_fusion_model

# ------------------------------------------------
# Dummy embeddings (replace with saved embeddings if you have them)
# ------------------------------------------------
# Simulating: BERT (768) + GNN (128)
num_samples = 100

text_embeddings = np.random.rand(num_samples, 768).astype("float32")
gnn_embeddings = np.random.rand(num_samples, 128).astype("float32")

labels = np.random.randint(0, 2, size=(num_samples, 1))

# ------------------------------------------------
# Build fusion model
# ------------------------------------------------
fusion_model = build_fusion_model(
    text_dim=768,
    user_dim=128
)

fusion_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------
# Train (FAST, CPU-friendly)
# ------------------------------------------------
fusion_model.fit(
    [text_embeddings, gnn_embeddings],
    labels,
    epochs=3,
    batch_size=8
)

# ------------------------------------------------
# SAVE MODEL (THIS IS WHAT YOU WERE MISSING)
# ------------------------------------------------
fusion_model.save("fusion_model_gnn.h5")

print("✅ Fusion model saved as fusion_model_gnn.h5")
