"""Merge the original repository index with the B handoff."""
import csv
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fields = "model_setting split sample_id npz_path gt_path mask_key embedding_storage".split()
def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected header in {path}")
        return list(reader)
def groups(rows, settings):
    sets = defaultdict(set)
    for row in rows:
        model, split, sample = (row[k] for k in ("model_setting", "split", "sample_id"))
        if model not in settings or split not in {"val", "test"} or sample in sets[model, split]:
            raise ValueError(f"Unexpected or duplicate row: {(model, split, sample)}")
        sets[model, split].add(sample)
    for split in ("val", "test"):
        pairs = [sets[m, split] for m in settings]
        if len(pairs[0]) != 150 or any(p != pairs[0] for p in pairs):
            raise ValueError(f"Models must share the same 150 {split} samples")
    return sets
path = root / "indices/all_samples.csv"
a_models = {"original", "linear_probe", "adapter"}
b_models = {"lora_r8", "lora_r16"}
current = read(path)
a = [r for r in current if r["model_setting"] not in b_models]
b = read(root / "indices/b_samples.csv")
groups(a+b,a_models | b_models)
temp = path.with_suffix(".tmp")
with temp.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(a+b)
temp.replace(path)
print(f"{len(a)} A+C rows + {len(b)} B rows = {len(a+b)}")
