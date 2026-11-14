"""Fail explicitly on invalid Adapter tensors instead of exporting empty masks."""

import math

import torch


def amp_dtype_from_config(config, device):
    name = config.get("amp_dtype", "float16")
    if name not in ("float16", "bfloat16"):
        raise ValueError("amp_dtype must be float16 or bfloat16")
    if config.get("amp") and device.type == "cuda" and name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support BF16 AMP")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


class NumericalError(RuntimeError):
    def __init__(self, message, **details):
        self.details = {"message": message, **details}
        super().__init__(f"{message}; details={details}")


def require_finite(tensor, name, **context):
    if not bool(torch.isfinite(tensor).all()):
        raise NumericalError(
            f"NaN/Inf detected in {name}",
            tensor=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            nonfinite_elements=int((~torch.isfinite(tensor)).sum().item()),
            **context,
        )


def nonfinite_tensor_names(tensors):
    """Synchronize once per device in the normal, finite case."""
    groups = {}
    for name, value in tensors.items():
        if torch.is_tensor(value):
            groups.setdefault(value.device, []).append((name, value))
    invalid = []
    for group in groups.values():
        flags = torch.stack([torch.isfinite(value).all() for _, value in group])
        if not bool(flags.all()):
            invalid.extend(name for (name, _), flag in zip(group, flags.tolist()) if not flag)
    return invalid


def require_finite_tensors(tensors, name, **context):
    invalid = nonfinite_tensor_names(tensors)
    if invalid:
        raise NumericalError(
            f"NaN/Inf detected in {name}",
            invalid_tensor_count=len(invalid),
            invalid_tensors=invalid[:12],
            **context,
        )


def optimizer_tensors(state):
    return {
        f"{parameter_id}.{key}": value
        for parameter_id, values in state.get("state", {}).items()
        for key, value in values.items()
        if torch.is_tensor(value)
    }


def require_valid_scale(scale, **context):
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise NumericalError("AMP gradient scale must be finite and positive", scale=scale, **context)


def validate_checkpoint(payload, training=False):
    if not payload.get("adapter_state"):
        raise ValueError("Checkpoint has no Adapter parameters")
    require_finite_tensors(payload["adapter_state"], "checkpoint Adapter parameters")
    if training:
        if "optimizer_state" not in payload:
            raise ValueError("Checkpoint has no optimizer state for training resume")
        require_finite_tensors(
            optimizer_tensors(payload["optimizer_state"]), "checkpoint optimizer state"
        )
        scaler_state = payload.get("scaler_state", {})
        if "scale" in scaler_state:
            require_valid_scale(scaler_state["scale"], phase="checkpoint validation")
        if int(payload.get("epoch", -1)) < 0:
            raise ValueError("Invalid completed epoch count in checkpoint")
        if not math.isfinite(float(payload.get("best_val_dice", float("nan")))):
            raise ValueError("Invalid best_val_dice in checkpoint")
