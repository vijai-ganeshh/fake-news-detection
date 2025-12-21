import tensorflow as tf

def run_feature_ablation(model, text_emb, user_emb):
    """
    Perform feature ablation on a fusion model by removing
    text or user embeddings and observing prediction changes.
    """

    # Full prediction
    full_pred = model([text_emb, user_emb], training=False).numpy()

    # Remove text
    no_text_pred = model(
        [tf.zeros_like(text_emb), user_emb],
        training=False
    ).numpy()

    # Remove user
    no_user_pred = model(
        [text_emb, tf.zeros_like(user_emb)],
        training=False
    ).numpy()

    return {
        "full_prediction": full_pred,
        "no_text_prediction": no_text_pred,
        "no_user_prediction": no_user_pred,
        "text_contribution": full_pred - no_text_pred,
        "user_contribution": full_pred - no_user_pred
    }
