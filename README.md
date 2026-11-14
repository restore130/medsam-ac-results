# MedSAM Linear Probe 与 Adapter 实验交付

本仓库提供代码、配置、训练曲线和汇总表。完整数据、基础权重、微调权重、900份预测及特征位于 [Release 下载页](https://github.com/restore130/medsam-ac-results/releases/tag/v1.0.0)。

## 下载完整交付包

下载 Release 中全部三个 `AC-delivery.zip.part001`、`part002`、`part003` 和 `SHA256SUMS.txt`，放在同一文件夹。不要分别解压分卷，先按顺序合并为 `AC-delivery.zip`。

Windows PowerShell：
```powershell
python -c "from pathlib import Path; import shutil; parts=sorted(Path('.').glob('AC-delivery.zip.part*')); assert len(parts)==3; out=open('AC-delivery.zip','wb'); [shutil.copyfileobj(p.open('rb'),out) for p in parts]; out.close()"
Get-FileHash AC-delivery.zip -Algorithm SHA256
```

Linux/macOS：
```bash
cat AC-delivery.zip.part001 AC-delivery.zip.part002 AC-delivery.zip.part003 > AC-delivery.zip
sha256sum AC-delivery.zip
```

与 SHA256SUMS.txt 中 AC-delivery.zip 的校验值核对后再解压。完整目录以解压得到的交付包为准；本代码仓库省略大体积数据、权重及预测文件，不可单独运行全量复现。原始数据和第三方 SAM 代码应遵守各自来源许可；本仓库不额外授予这些第三方内容的许可。

---

# MedSAM 训练组交付说明

本包交付原始 MedSAM、Linear Probe、Adapter 的模型代码、权重、训练记录、预测概率和掩码嵌入。数据集为 Kvasir-SEG，固定 train/val/test=700/150/150，种子 42，使用真值紧致 bbox。输入沿用第一组 1024×1024 预处理。预测阈值为 probability>0.5，评价在原图尺寸逐样本计算。

## 从哪里开始

1. 第三组读取 `indices/all_samples.csv`，按 model_setting、split、sample_id 定位概率和 Mask；GT 路径也已列出。
2. 第四组使用相同索引读取 `mask_embedding`，形状为 32×256×256，来自 mask decoder 的 output_upscaling。原始特征未归一化、未去均值。原图 GT 对应到特征网格时使用最近邻变换。
3. 使用 `scripts/load_result.py` 的 load_result 函数获得统一字段，避免分别适配 A 的 mask 与 C 的 prediction_mask。依赖 numpy；所有路径相对本包根目录解析。
4. 查看 `comparison/` 的 Dice、资源和范数统计表，以及 `verification/` 的检查结果。

读取示例（在交付包根目录）：
```python
from scripts.load_result import load_result
import csv
with open('indices/all_samples.csv', encoding='utf-8-sig') as f:
    row = next(csv.DictReader(f))
r = load_result(row['model_setting'], row['split'], row['sample_id'])
print(r['probability'].shape, r['prediction_mask'].shape, r['mask_embedding'].shape)
```

## 模型与结果位置

| 设置 | 方法目录 | 最佳权重 | 预测与特征 |
|---|---|---|---|
| 原始模型 | methods/C | weights/medsam_vit_b.pth，不微调 | methods/C/results/original/{val,test} |
| Linear Probe | methods/A | results/linear_probe/best.pth，第3轮 | methods/A/results/linear_probe/{val,test} |
| Adapter | methods/C | checkpoints/adapter_bf16/best.pt，第11轮 | methods/C/results/adapter_bf16/{val,test} |

权重路径均相对于对应方法目录。Linear Probe best.pth 只包含输出头及训练状态，Adapter 权重按 C 的加载接口使用；两者都需要 MedSAM 原始权重，不能把微调权重单独当作完整模型。

## Linear Probe 复现

只训练 mask_decoder.output_hypernetworks_mlps，冻结其余模块；AdamW lr=1e-3、weight_decay=0.01，20 epoch、batch=1，FP16 AMP，Dice+BCE 等权相加，Dice smooth=1e-5，无增强或调度器。每轮完整 val，以均值 Dice 最高选 best。冻结图像编码特征缓存在 image_feature_cache_v1；更换输入、预处理、基础权重或启用随机增强时不能复用旧缓存。

训练环境为 RTX4090、PyTorch2.5.1+cu121；导出环境为 V100、PyTorch2.6.0+cu124。建议独立环境，按机器安装兼容 GPU 版 torch/torchvision，再安装 numpy、Pillow、scikit-image、matplotlib、segment-anything。SAM 可从本包 methods/C/vendor/segment_anything 安装。A 为 CUDA 入口，需要 GPU。

在 methods/A 目录执行导出：
```bash
python server_train_linear_probe.py --project . --output results/linear_probe --epochs 20 --export val
python server_train_linear_probe.py --project . --output results/linear_probe --epochs 20 --export test
```
上述命令会覆盖对应 A 结果，重现实验前请复制方法目录或备份 results。断点续训使用同入口加 --resume；当前20轮已经完成，不需继续训练。新的训练使用新的 output 目录。

## Adapter 和原始模型复现

详见 `methods/C/docs/复现步骤.md` 与 `methods/C/README.md`。保留独立环境，不把 C 的历史恢复训练误写成全程 BF16 新实验。C 导出入口拒绝非空输出目录，请用 reproduced 等新目录。V100 不支持 BF16；如需重现 C 的 BF16 推理，应使用支持 BF16 的 GPU，不能不记录差异就改成 FP16。

## 统计与交接

`comparison/embedding_norms_per_sample.csv` 保存每个样本的逐像素 L2 范数均值、标准差以及整个嵌入张量 L2；summary 按图等权平均，跨样本标准差 ddof=0。A 的 float16 特征读取时转 float32 计算，不能恢复存储时损失的精度。

原始模型不训练，因此没有训练 epoch 或训练曲线。资源表不同方法的计时范围不同，必须连同 scope 列使用，不作同硬件速度优劣结论。推理耗时见 Dice 汇总表和各 split summary。HD95、SegECE、错误像素 AUROC 由第三组统一实现；距离、漂移、SVD 和降维由第四组处理。不得用单一 Dice 推断校准或表征质量。

实际差异见 `comparison/实验差异与结果说明.md`。原始文件保留在 methods 内，文件中的历史绝对路径仅用于追溯，复现采用本包相对路径。
