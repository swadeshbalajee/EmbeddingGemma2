"""All embedding work happens OFFLINE. Encoder weights are never fine-tuned."""
import numpy as np
import torch


def make_encoder(model_id="google/embeddinggemma-2", device=None, only_text=False):
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    config = {"vision_config": None, "audio_config": None} if only_text else {}
    model = SentenceTransformer(model_id, device=device,
                                model_kwargs={"torch_dtype": dtype}, config_kwargs=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_text(model, texts, batch_size=32):
    # Explicit no task prompt; same semantic space/interface as unprefixed media.
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, prompt="", show_progress_bar=False)
    emb = np.asarray(emb, np.float32).reshape(len(texts), -1)
    if emb.shape[1] != 768 or not np.isfinite(emb).all():
        raise ValueError(f"Unexpected or invalid text embeddings: {emb.shape}")
    return emb


@torch.inference_mode()
def encode_media(model, kind, paths):
    out = []
    for path in paths:
        obj = {kind: str(path)}
        emb = model.encode(obj, convert_to_numpy=True, normalize_embeddings=True,
                           prompt="", show_progress_bar=False)
        arr = np.asarray(emb, np.float32).reshape(-1)
        if arr.size != 768 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {kind} embedding {path}: {arr.shape}")
        out.append(arr)
    return np.stack(out)
