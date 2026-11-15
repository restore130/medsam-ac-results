# 第二组人员 B · MedSAM LoRA-r8 交付

本包包含 20 轮训练、最佳 LoRA 权重、原始 MedSAM ViT-B 权重、Kvasir-SEG 数据和划分、训练与续训代码，以及验证集和测试集各 150 份导出。可按 sample_id 与共享划分表对齐。

## A、第三组与第四组

- A 汇总 runs/formal-r8 下的 best.pt、last.pt、history.csv、dice.png 和 run_config.json，以及 runs/formal-r8.log 和根目录 report-r8.json。日志保留了中断和重启记录。
- 第三组使用 val/、test/ 内的 PNG 或 NPZ 的 prediction，以及完整前景 probability。逐样本 Dice、IoU、HD95 和嵌入范数在各 split 的 metrics.csv，汇总在 summary.json。
- 第四组以 sample_id 为键读取 NPZ 的 mask_embedding。该数组是 mask_decoder.output_upscaling 的原始输出去掉 batch 维，float32，形状 (32, 256, 256)。每个空间位置在 32 个通道上求 L2 范数，再对空间位置统计均值、标准差和 5/50/95 分位数。report-r8.json 将这些逐样本统计分别按 split 取平均。

## NPZ 字段

- sample_id 为样本标识，对应 data/splits 下 CSV 的 sample_id。
- original_size 为 (H, W)。probability 和 prediction 都是原图分辨率，前者是 [0,1] 的前景概率图，后者是 0/1 二值掩码。PNG 使用 0/255。
- bbox_1024 按 (x_min, y_min, x_max, y_max) 保存 1024×1024 模型输入坐标中的提示框。换回原图时，x 乘 W/1024，y 乘 H/1024。

## 环境

建议使用独立的 Python 3.12 环境、Git 和兼容 CUDA 12.4 的 NVIDIA 驱动。在本目录的 PowerShell 中运行。

```powershell
python -m venv .venv
& ./.venv/Scripts/python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '1'
$py = './.venv/Scripts/python.exe'
```

## 重跑、续训与重新导出

训练命令写入新的结果目录。下面按实际的梯度检查点切换轮次重跑。第二条的 --resume 会读取该目录 last.pt 的优化器、AMP scaler 和历史。原 JSON 保留完整校验。可在第 14–20 轮的任意已保存 checkpoint 上重复同一续训命令。

```powershell
& $py -u train_lora.py --project MedsamLearning --config lora-r8.json --output runs/reproduction-r8 --stop-after-epoch 13
& $py -u train_lora.py --project MedsamLearning --config lora-r8.json --output runs/reproduction-r8 --resume --no-gradient-checkpointing
```

只用交付的最佳权重重新推理和导出两个 split，写入新的目录。

```powershell
& $py -u train_lora.py --project MedsamLearning --config lora-r8.json --output runs/reexport-r8 --export-only runs/formal-r8/best.pt
```

## 实际训练与资源

- 第 1–2 轮在 RTX 3060 Laptop GPU 上训练。原设备记录见 provenance/epochs-01-02-run_config.json。
- 第 3 轮在 RTX 3090 上训练，仍启用梯度检查点。该轮保存了 checkpoint，绘图时因 OpenMP 冲突退出。
- 第 4 轮起使用隔离环境。NumPy 仍为 1.26.4，从 Anaconda 的 MKL 构建换为 PyPI 构建，SciPy 换为 1.15.3（旧版本未记录），scikit-image 从 0.24.0 更新到 0.25.2。PyTorch 保持 2.5.1+cu124。完整可安装依赖见 requirements.txt。
- 第 4–13 轮在 RTX 3090 上启用梯度检查点。第 14–20 轮在同一块 RTX 3090 上关闭梯度检查点。AMP 全程开启。LoRA 注入 12 个编码器块的 Q/V，rank=8，alpha=8，可训练参数 294,912。
- report-r8.json 按阶段记录各设备的训练和验证耗时及峰值显存。耗时是已保存轮次的记录之和。峰值是 PyTorch 训练阶段分配的字节数。最终导出耗时单列在各 split 的 summary.json。换机和依赖变化可能影响浮点复现。
- best.pt 来自第 3 轮。训练期间的验证 Dice 约 0.963416；用该权重最终重推理的验证 Dice 为 0.963461。测试集 Dice 为 0.965030。
