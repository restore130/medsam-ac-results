import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize

TARGET_SIZE = 1024
MASK_THRESHOLD = 127


def read_split(project_dir, split):
    """读取已有划分，不重新随机划分。"""
    csv_path = Path(project_dir) / "data" / "splits" / f"{split}.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def prepare_sample(project_dir, row):
    """所有划分共用：读取图像、处理 Mask、生成框并预处理。"""
    project_dir = Path(project_dir)
    with Image.open(project_dir / row["image_path"]) as img:
        image_rgb = np.asarray(img.convert("RGB"))
    with Image.open(project_dir / row["mask_path"]) as img:
        gt_mask = (np.asarray(img.convert("L")) > MASK_THRESHOLD).astype(np.uint8)

    height, width = image_rgb.shape[:2]
    if gt_mask.shape != (height, width):
        raise ValueError("图像与 Mask 的原始尺寸不一致")

    # 原始 Mask 全部前景的共同紧致框；不添加随机扰动。
    ys, xs = np.where(gt_mask == 1)
    if xs.size == 0:
        raise ValueError("Mask 没有前景，无法生成 bbox")
    bbox_original = np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
    )

    # 沿用 07 中已经检查过的图像处理流程。
    image_resized = resize(
        image_rgb, (TARGET_SIZE, TARGET_SIZE), order=3,
        preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    pixels = image_resized.astype(np.float32)
    minimum, maximum = float(pixels.min()), float(pixels.max())
    normalized = (pixels - minimum) / max(maximum - minimum, 1e-8)
    image_chw = np.ascontiguousarray(
        normalized.transpose(2, 0, 1), dtype=np.float32
    )

    # Mask 用最近邻缩放；原始 Mask 也保留，供后续评价使用。
    gt_mask_1024 = resize(
        gt_mask, (TARGET_SIZE, TARGET_SIZE), order=0,
        preserve_range=True, anti_aliasing=False
    ).astype(np.uint8)

    # 除数是 [宽, 高, 宽, 高]；坐标保留浮点数。
    bbox_1024 = (
        bbox_original
        / np.array([width, height, width, height], dtype=np.float32)
        * TARGET_SIZE
    )
    return {
        "sample_id": row["sample_id"],
        "image": image_chw,
        "gt_mask": gt_mask,
        "gt_mask_1024": gt_mask_1024,
        "bbox_original": bbox_original,
        "bbox_1024": bbox_1024,
        "original_size": np.array([height, width], dtype=np.int64),
    }
