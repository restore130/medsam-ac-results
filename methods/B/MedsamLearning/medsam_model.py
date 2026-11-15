"""MedSAM model loading and box-prompt inference helpers."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

MEDSAM_VIT_B_MD5 = "3bb6db55bd0c9ca30b61248bca72f8d6"


def resolve_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_md5(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_medsam(checkpoint, device="auto", verify_checksum=False):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"MedSAM checkpoint not found: {checkpoint}. Run download_weights.py first."
        )
    if verify_checksum:
        actual = file_md5(checkpoint)
        if actual != MEDSAM_VIT_B_MD5:
            raise ValueError(
                f"checkpoint MD5 mismatch: expected {MEDSAM_VIT_B_MD5}, got {actual}"
            )
    try:
        from segment_anything import sam_model_registry
    except ImportError as error:
        raise RuntimeError(
            "segment_anything is not installed; install requirements-model.txt"
        ) from error

    target = resolve_device(device)
    # The official MedSAM checkpoint contains CUDA-backed tensors. Building SAM
    # with checkpoint=... calls torch.load without map_location and therefore
    # fails in CPU-only environments. Load on CPU explicitly, then move once.
    model = sam_model_registry["vit_b"](checkpoint=None)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.to(target)
    model.eval()
    return model, target


@torch.inference_mode()
def predict_from_boxes(model, images, boxes_1024, output_size, use_amp=False):
    """Return binary masks, probabilities and image embeddings for one batch."""
    amp_enabled = bool(use_amp and images.device.type == "cuda")
    with torch.autocast(device_type=images.device.type, enabled=amp_enabled):
        image_embeddings = model.image_encoder(images)
        boxes = boxes_1024.to(dtype=torch.float32, device=images.device)
        if boxes.ndim == 2:
            boxes = boxes[:, None, :]
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res_logits, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        probabilities = torch.sigmoid(low_res_logits)
        probabilities = F.interpolate(
            probabilities, size=tuple(output_size), mode="bilinear", align_corners=False
        )
    masks = probabilities > 0.5
    return masks, probabilities, image_embeddings
