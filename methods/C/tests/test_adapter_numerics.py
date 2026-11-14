import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from adapter_model import BottleneckAdapter, adapter_state_dict, load_adapter_state_dict
from adapter_numerics import NumericalError, require_finite, validate_checkpoint
from train_adapter import load_resume, save_checkpoint, set_seed, train_one_epoch


class TinyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        block = nn.Module()
        block.attention_adapter = BottleneckAdapter(4, 2)
        self.image_encoder.blocks = nn.ModuleList([block])


class NumericalSafetyTests(unittest.TestCase):
    def test_logits_are_checked_before_sigmoid_can_hide_infinity(self):
        with self.assertRaises(NumericalError):
            require_finite(torch.tensor([float("inf")]), "logits", sample_id="example")

    def test_corrupt_adapter_state_is_rejected_without_partial_loading(self):
        model = TinyAdapterModel()
        original = adapter_state_dict(model)
        corrupt = {name: value + 1 for name, value in original.items()}
        corrupt[next(reversed(corrupt))].flatten()[0] = float("nan")
        with self.assertRaises(NumericalError):
            load_adapter_state_dict(model, corrupt)
        for name, value in adapter_state_dict(model).items():
            self.assertTrue(torch.equal(value, original[name]))

    def test_corrupt_logits_stop_before_any_optimizer_step(self):
        model = TinyAdapterModel()
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        batch = {
            "sample_id": ["bad_sample"],
            "image": torch.zeros(1, 3, 4, 4),
            "bbox_1024": torch.ones(1, 4),
            "gt_mask_1024": torch.ones(1, 4, 4),
        }
        with patch("train_adapter.forward_logits", return_value=torch.full((1, 1, 4, 4), float("nan"))):
            with patch.object(optimizer, "step") as step:
                with self.assertRaises(NumericalError) as caught:
                    train_one_epoch(model, [batch], optimizer, scaler, torch.device("cpu"), False, 1e-6)
                step.assert_not_called()
        self.assertEqual(caught.exception.details["sample_id"], "bad_sample")

    def test_checkpoint_rejects_corrupt_optimizer_or_zero_scale(self):
        payload = {"adapter_state": {"weight": torch.ones(1)}, "epoch": 8,
                   "best_val_dice": 0.9, "scaler_state": {},
                   "optimizer_state": {"state": {0: {"exp_avg": torch.tensor([float("nan")])}}}}
        with self.assertRaises(NumericalError):
            validate_checkpoint(payload, training=True)
        payload["optimizer_state"]["state"] = {}
        payload["scaler_state"] = {"scale": 0.0}
        with self.assertRaises(NumericalError):
            validate_checkpoint(payload, training=True)

    def test_resume_uses_next_epoch_and_failed_save_preserves_good_file(self):
        set_seed(42)
        model = TinyAdapterModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        original = adapter_state_dict(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, scaler, 8, 0.9, {"bottleneck_dimension": 2})
            good_bytes = path.read_bytes()
            with torch.no_grad():
                next(model.parameters()).flatten()[0] = float("nan")
            with self.assertRaises(NumericalError):
                save_checkpoint(path, model, optimizer, scaler, 9, 0.9, {"bottleneck_dimension": 2})
            self.assertEqual(path.read_bytes(), good_bytes)
            start_epoch, best_dice = load_resume(path, model, optimizer, scaler, torch.device("cpu"))
        self.assertEqual(start_epoch, 8)
        self.assertEqual(list(range(start_epoch, 20))[0] + 1, 9)
        self.assertEqual(best_dice, 0.9)
        for name, value in adapter_state_dict(model).items():
            self.assertTrue(torch.equal(value, original[name]))


if __name__ == "__main__":
    unittest.main()
