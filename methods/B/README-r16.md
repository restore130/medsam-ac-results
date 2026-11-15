# MedSAM LoRA-r16

环境安装、数据及 NPZ 字段与 README-r8.md 相同。r16 使用 lora-r16.json，rank=16，alpha=16，可训练参数 589,824。AMP 全程开启。

## 训练、续训与重新导出

在根目录按 README-r8.md 设置 $py 后执行。两条训练命令按实际梯度检查点切换轮次重跑，输出到新目录。

```powershell
& $py -u train_lora.py --project MedsamLearning --config lora-r16.json --output runs/reproduction-r16 --stop-after-epoch 12
& $py -u train_lora.py --project MedsamLearning --config lora-r16.json --output runs/reproduction-r16 --resume --no-gradient-checkpointing
& $py -u train_lora.py --project MedsamLearning --config lora-r16.json --output runs/reexport-r16 --export-only runs/formal-r16/best.pt
```

从中间某轮续训时，把相应的 last.pt 放入续训目录，并按该轮的实际设置选用梯度检查点开关。交付中的 last.pt 是第 20 轮。

## 设备与资源出处

- 第 1–12 轮为外机训练。迁移包的 run_config.json 记录 GPU 为 NVIDIA GeForce RTX 3060 Laptop GPU，PyTorch 为 2.5.1+cu124。逐轮训练、验证耗时及 PyTorch 分配显存峰值来自迁移包的 history.csv。原始记录均在 provenance/。
- 第 13–20 轮在本机 RTX 3090 上续训，关闭梯度检查点，沿用隔离环境的依赖。耗时和峰值取自本机保存的逐轮记录。
- report-r16.json 按阶段汇总耗时和峰值，按 split 汇总最终评测和嵌入范数统计。训练日志前段来自外机迁移包，后段来自本机续训。
- 待原机器补齐（若有记录）：第 1–12 轮各次启动的设备映射与驱动版本、Python 和完整依赖版本、原机数据划分哈希、整卡显存和端到端耗时。迁移包提供的是模型训练进程的逐轮分配峰值，以及训练和验证阶段耗时。结果样本 ID 已按本机随包交付的划分核验。
