import torch
import numpy as np

from text_model.bert_encoder import get_text_embeddings
from gnn_model.gnn import GNNModel
from fusion_model.fusion_classifier import FusionClassifier


device = torch.device("cpu")


# ---------------- GNN ----------------
gnn_model = GNNModel(
    input_dim=768,
    hidden_dim=256
)
gnn_model.load_state_dict(
    torch.load("gnn_model.pt", map_location=device)
)
gnn_model.to(device)
gnn_model.eval()


# ---------------- Fusion ----------------
fusion_model = FusionClassifier(input_dim=960)
fusion_model.load_state_dict(
    torch.load("fusion_model_gnn.pt", map_location=device)
)
fusion_model.to(device)
fusion_model.eval()


def predict_news(text: str) -> float:
    text_emb = get_text_embeddings([text])
    text_emb = np.array(text_emb, dtype=np.float32)
    assert text_emb.shape == (1, 768)

    text_tensor = torch.from_numpy(text_emb).to(device)

    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    user_features = torch.zeros((1, 64), device=device)

    with torch.no_grad():
        gnn_emb = gnn_model(text_tensor, edge_index)
        assert gnn_emb.shape == (1, 128)

        fusion_input = torch.cat(
            [text_tensor, gnn_emb, user_features],
            dim=1
        )
        assert fusion_input.shape == (1, 960)

        output = fusion_model(fusion_input)

    return float(output.item())


if __name__ == "__main__":
    sample_text = "Breaking news: scientists confirm water is wet"
    print("Fake news probability:", predict_news(sample_text))
