# inference/predict.py

import torch
import numpy as np

# -------- Text Encoder (TensorFlow) --------
from text_model.bert_encoder import get_text_embeddings

# -------- PyTorch Models --------
from gnn_model.gnn import GNNModel
from fusion_model.fusion_classifier import FusionClassifier


# =====================================================
# Device
# =====================================================
device = torch.device("cpu")


# =====================================================
# Load GNN model (MUST match trained architecture)
# Trained checkpoint shows:
#   conv1: 768 -> 256
#   conv2: 256 -> 128
# =====================================================
gnn_model = GNNModel(
    input_dim=768,
    hidden_dim=256   # IMPORTANT: must be 256
)

gnn_model.load_state_dict(
    torch.load("gnn_model.pt", map_location=device)
)
gnn_model.eval()


# =====================================================
# Load Fusion model (PyTorch)
# Input = text(768) + gnn(128) = 896
# =====================================================
fusion_model = FusionClassifier(
    input_dim=768 + 128
)

fusion_model.load_state_dict(
    torch.load("fusion_model_gnn.pt", map_location=device)
)
fusion_model.eval()


# =====================================================
# Prediction Function
# =====================================================
def predict_news(text: str) -> float:
    """
    Predict fake/real probability for a news text.
    Returns probability between 0 and 1.
    """

    # -------------------------------------------------
    # 1. Get BERT embeddings (TensorFlow → NumPy)
    # -------------------------------------------------
    text_emb = get_text_embeddings([text])  # shape: (1, 768)
    text_emb = np.array(text_emb, dtype=np.float32)

    # -------------------------------------------------
    # 2. Convert to Torch tensor
    # -------------------------------------------------
    text_tensor = torch.from_numpy(text_emb).to(device)

    # -------------------------------------------------
    # 3. Dummy graph (single node, self-loop)
    # -------------------------------------------------
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    with torch.no_grad():
        # GNN embedding → (1, 128)
        gnn_emb = gnn_model(text_tensor, edge_index)

        # Fusion input → (1, 896)
        fusion_input = torch.cat([text_tensor, gnn_emb], dim=1)

        # Prediction
        output = fusion_model(fusion_input)

    return float(output.item())
