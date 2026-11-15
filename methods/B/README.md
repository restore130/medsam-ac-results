# 第二组人员 B · MedSAM LoRA-r8 / LoRA-r16 交付

两档共用一份原始 MedSAM ViT-B 权重、Kvasir-SEG 数据及划分、训练与推理导出代码。两档的 Q/V 注入位置和训练规则相同，alpha 随 rank 调整，alpha/r 均为 1。

- A 汇总 runs/formal-r8、runs/formal-r16、两份训练日志和 report-r8.json、report-r16.json。
- 第三组使用各档 val/、test/ 内的 PNG 或 NPZ prediction 和完整 probability，按 sample_id 对齐。
- 第四组使用各档 NPZ 的 mask_embedding，按 sample_id 对齐。每档各有 val 150 和 test 150 份。
- 环境安装、NPZ 字段、指标与阶段记录见 README-r8.md 和 README-r16.md。
