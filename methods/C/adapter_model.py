"""Bottleneck adapters for the official MedSAM ViT-B image encoder."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from adapter_numerics import require_finite_tensors, validate_checkpoint
from medsam_model import load_medsam, resolve_device


class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter with PyTorch's default Linear initialization."""

    def __init__(self, dimension, bottleneck_dimension=16):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck_dimension)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dimension, dimension)

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))


class AdaptedViTBlock(nn.Module):
    """Wrap a SAM ViT block and add adapters after both native residual paths."""

    def __init__(self, block, bottleneck_dimension=16):
        super().__init__()
        self.block = block
        self.gradient_checkpointing = False
        dimension = block.norm1.normalized_shape[0]
        self.attention_adapter = BottleneckAdapter(dimension, bottleneck_dimension)
        self.mlp_adapter = BottleneckAdapter(dimension, bottleneck_dimension)

    def _forward_impl(self, x):
        # Keep the upstream SAM block's windowed-attention path unchanged.
        from segment_anything.modeling.image_encoder import (
            window_partition,
            window_unpartition,
        )

        shortcut = x
        attention_input = self.block.norm1(x)
        if self.block.window_size > 0:
            height, width = attention_input.shape[1], attention_input.shape[2]
            attention_input, pad_hw = window_partition(
                attention_input, self.block.window_size
            )

        attention_output = self.block.attn(attention_input)
        if self.block.window_size > 0:
            attention_output = window_unpartition(
                attention_output, self.block.window_size, pad_hw, (height, width)
            )

        x = shortcut + attention_output
        x = self.attention_adapter(x)
        x = x + self.block.mlp(self.block.norm2(x))
        x = self.mlp_adapter(x)
        return x

    def forward(self, x):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


def inject_adapters(model, bottleneck_dimension=16):
    """Freeze MedSAM and wrap every ViT block with two trainable adapters."""
    model.requires_grad_(False)
    encoder = model.image_encoder
    encoder.blocks = nn.ModuleList(
        AdaptedViTBlock(block, bottleneck_dimension)
        for block in encoder.blocks
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("attention_adapter" in name or "mlp_adapter" in name)
    model.eval()
    return model


def set_gradient_checkpointing(model, enabled=True):
    for block in model.image_encoder.blocks:
        block.gradient_checkpointing = bool(enabled)


def adapter_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def adapter_state_dict(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in adapter_parameters(model).items()
    }


def load_adapter_state_dict(model, state):
    parameters = adapter_parameters(model)
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(f"Adapter keys mismatch; missing={missing}, unexpected={unexpected}")
    for name, parameter in parameters.items():
        value = state[name]
        if parameter.shape != value.shape:
            raise ValueError(
                f"Adapter shape mismatch for {name}: expected {tuple(parameter.shape)}, "
                f"got {tuple(value.shape)}"
            )
    # Validate the entire state before changing any parameter.
    require_finite_tensors(state, "Adapter parameters to load")
    for name, parameter in parameters.items():
        value = state[name]
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def build_adapter_model(
    checkpoint,
    device="auto",
    bottleneck_dimension=16,
    verify_checksum=True,
):
    """Load the original checkpoint first, inject adapters, then move the model."""
    target = resolve_device(device)
    model, _ = load_medsam(
        Path(checkpoint), device="cpu", verify_checksum=verify_checksum
    )
    inject_adapters(model, bottleneck_dimension=bottleneck_dimension)
    model.to(target)
    return model, target


def load_adapter_checkpoint(model, path, map_location="cpu"):
    try:
        # Checkpoints are produced locally by train_adapter.py and include RNG
        # state for exact resume, which is not a weights-only payload.
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location=map_location)
    if "adapter_state" not in payload:
        raise ValueError(f"Not an Adapter checkpoint: {path}")
    validate_checkpoint(payload)
    load_adapter_state_dict(model, payload["adapter_state"])
    return payload
