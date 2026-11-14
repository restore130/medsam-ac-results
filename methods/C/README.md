# 第二组原始 MedSAM 与 Adapter 个人交付

本包覆盖本人两项任务：原始 MedSAM 不微调对照；Adapter 独立训练、推理与导出。依据 reference/项目说明 中的实验指导书和分工文档整理。Linear Probe 的配套交付位于本整合包 methods/A。

先读 docs/交付对照表.md，再看 docs/结果与运行记录.md、docs/字段与交接说明.md。运行命令见 docs/复现步骤.md。

- 根目录 Python：模型加载、训练、推理、导出与公共数据接口。
- config：实际 BF16 配置、前期 FP16 配置、原始模型推理参数。
- weights：官方 MedSAM 基座，只保存一份。
- checkpoints：Adapter 第 11 轮 best、第 20 轮 last、健康第 8 轮恢复权重。
- results/original/{val,test}：原始模型各 150 NPZ。
- results/adapter_bf16/{val,test}：Adapter 各 150 NPZ；train 有训练日志。
- tables、figures：分割参考指标、有效训练记录与曲线、资源参数、嵌入范数统计与分布图。
- indices：600 条结果索引、样本 ID 和 GT 路径。
- data：第一组原始数据、manifest、固定划分，作为复现附件，不认作本人新完成的数据成果。
- logs：失败阶段日志及当时代码，不含损坏权重和冒烟结果。
- environment、vendor：环境快照、SAM 源码、许可证及来源。
- verification：检查记录；scripts/verify_delivery.py 检查整个交付包。

所有文件都是独立复制，源目录与旧交付包保留。没有复制虚拟环境。历史 summary 的绝对路径只作溯源，复现使用相对路径。

安装后在本包根目录运行 python scripts/verify_delivery.py。SHA256SUMS.csv 不包括自身、PACKAGING.json、verification/package_check.json、Python 缓存。新生成的 reproduced 不属于原始交付清单。
