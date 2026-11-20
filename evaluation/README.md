# 五组模型像素级不确定性评测 —— 结论与结果

评测对象：original / linear_probe / adapter / lora_r8 / lora_r16，统一口径。

## 结论

1. 分得准不准（Dice）：original 明显最差（val 0.909 / test 0.936）；四组微调接近，
   adapter 与 lora_r16 略优（0.965~0.966）。
2. 概率是否校准（SegECE）：original 校准明显最差（0.0368，是其余四组的 4~7 倍），
   微调方法的置信度显著更校准。
3. 高熵能否定位错误（AUROC）：original 最低（0.946），其余约 0.98，
   微调模型的高熵区域更可靠地指向真实错误像素。

## 结果摘要（val / test）

| 模型 | Dice | SegECE（图均） | 像素 AUROC |
|---|---|---|---|
| original | 0.9086 / 0.9362 | 0.0368 / 0.0216 | 0.9458 / 0.9672 |
| linear_probe | 0.9531 / 0.9616 | 0.0096 / 0.0074 | 0.9796 / 0.9807 |
| adapter | 0.9656 / 0.9654 | 0.0053 / 0.0061 | 0.9827 / 0.9859 |
| lora_r8 | 0.9635 / 0.9650 | 0.0053 / 0.0062 | 0.9840 / 0.9849 |
| lora_r16 | 0.9660 / 0.9616 | 0.0057 / 0.0089 | 0.9829 / 0.9827 |

HD95（原图像素）同样：original 47.4 / 32.1 最差，adapter 19.0 / 17.6 最好，其余居中。

## 文件清单（要引用哪个看这里）

| 文件 | 内容 |
|---|---|
| `summary_metrics.csv` | 五组 × val/test 汇总（Dice / IoU / HD95 / SegECE / AUROC） |
| `per_sample_metrics.csv` | 1500 行逐图指标 |
| `reliability_curves.csv` | 可靠性曲线分箱数据 |
| `figures/comparison_segece.png` | SegECE 柱状图（val + test） |
| `figures/comparison_auroc.png` | 像素 AUROC 柱状图（val + test） |
| `figures/dice_segece_scatter.png` | Dice–SegECE 散点 |
| `figures/reliability_val.png` / `reliability_test.png` | 可靠性曲线 |
| `figures/case_*.png` | 3 张典型病例图（high / low / boundary） |

## 指标口径

- 熵 `H(p) = -p·log p - (1-p)·log(1-p)`，画图除以 `log2` 归一到 0–1。
- SegECE：置信度 `max(p, 1-p)`，固定 10 箱（五组共用相同箱边界），
  报告采用「逐图算再等权平均」口径（`summary_metrics.csv` 的 `seg_ece_per_image_mean`）。
- 像素 AUROC：标签 = 错误像素，分数 = 熵，逐图算再等权平均。
- HD95：对称表面距离的双向 95 分位取最大值，原图像素单位。
- 概率统一用保存的 `probability`（是前景概率，**不是 logits**）。

## 病例选取规则

high = 五组平均 Dice 最高；low = 五组平均 Dice 最低；
boundary = original 错误像素落在 GT 边界 5px 内比例最高（≥1000 错误像素）。

## 写报告必带的限定

1. 用**真值 bbox** 做统一提示，不等同无真值框的实际部署。
2. 各方法训练精度 / 硬件 / 实现有差异，校准差异**不能全归因于 PEFT 方法本身**。
3. Dice（准不准）、SegECE（校准）、AUROC（高熵定位错误）三者不可互相替代。

## 复现

脚本在 `uncertainty_metrics.py` / `plot_cases.py`，数据在训练组 Release（见根目录 `README.md`）。
需要重跑时，把两个脚本复制到解压根目录运行即可。
