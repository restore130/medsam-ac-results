# MedSAM LoRA 人员 B 最终交付

将核心 ZIP 和四个结果 ZIP 解压到同一目录。完整包包含原始 Kvasir-SEG 数据、MedSAM 基础权重、r8/r16 最佳和续训权重、20 轮日志与曲线、600 份 NPZ 及对应 PNG。数据划分 700/150/150，种子 42，使用真值紧致框。预测采用原图概率 > 0.5。第三组和第四组可通过 indices/all_samples.csv 和 scripts/load_result.py 读取结果。

从解压根目录复现导出（使用 Windows PowerShell，其他系统可改写路径分隔符）。安装兼容 CUDA 的 PyTorch/torchvision 后，使用 methods/B/requirements.txt 安装依赖：

    python methods/B/train_lora.py --project methods/B/MedsamLearning --config methods/B/lora-r8.json --output reproduced/r8 --export-only methods/B/runs/formal-r8/best.pt
    python methods/B/train_lora.py --project methods/B/MedsamLearning --config methods/B/lora-r16.json --output reproduced/r16 --export-only methods/B/runs/formal-r16/best.pt

两档复现训练和换机命令见 methods/B/README-r8.md 与 methods/B/README-r16.md。复现使用新输出目录。训练、换机和历史环境见两档说明及 provenance/。训练期最佳验证 Dice 与最终重新导出的验证 Dice 分开记录。跨设备资源数据记录各自条件。

comparison/B_dice_summary.csv 的 HD95 是 B 导出脚本算出的原图像素值，第三组以统一实现为正式口径，并负责 SegECE、错误像素 AUROC。第四组负责跨设置表征比较。B 的范数统计包括逐像素 32 维 L2 的图内均值和标准差，以及整张特征的 L2。跨样本按图等权，标准差 ddof=0。嵌入以 float32 原始值交付，未归一化或去均值。

GitHub 提交包保留源码、曲线、汇总和 B 的索引。仓库根目录已有 A+C 索引时，从仓库根目录执行：

    python scripts/merge_b_index.py

脚本校验 900 条 A+C 和 600 条 B 的样本对齐，再保留原有记录并追加至 1,500 条。仓库原有 load_result.py 按 mask_key 解析预测字段，B 对应 prediction。完整附件和 SHA256 见 [B Release](https://github.com/restore130/medsam-ac-results/releases/tag/b-lora-v1.0.0)。数据及第三方代码遵循各自来源许可。


Kvasir-SEG 引用 Jha 等，*Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020。该数据用于科研与教育。参见[数据来源及使用条款](https://datasets.simula.no/kvasir-seg/)和[MedSAM 基础权重](https://zenodo.org/records/10689643)。
