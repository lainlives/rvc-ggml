# tools/ —— 模型转换与校验工具（Python）

一次性转换/校验工具，**不是训练运行期依赖**（训练/预处理/导出全 C++）。

| 脚本 | 作用 |
|---|---|
| `pth_to_train_gguf.py` | 官方 RVC `.pth`（预训练 G/D 或 WebUI 提取的推理小模型）→ 训练态 GGUF（微调/续训的输入格式） |
| `verify_train_gguf.py` | 训练态 GGUF 结果校验：与源 `.pth` 的 state_dict 逐字节对比，或校验 C++ 装载哈希 |

## 依赖

- Python 3.9+（仅这两个工具需要 `torch`；训练本体无需 Python）

## 典型流程（微调官方预训练模型）

```powershell
# 1. 官方预训练权重 → 训练态 GGUF
python tools\pth_to_train_gguf.py --pth <RVC目录>\assets\pretrained\f0G40k.pth --config configs\v1-40k-f0.json --out work\f0G40k.train.gguf
python tools\pth_to_train_gguf.py --pth <RVC目录>\assets\pretrained\f0D40k.pth --config configs\v1-40k-f0.json --out work\f0D40k.train.gguf

# 2.（可选）校验转换结果
python tools\verify_train_gguf.py --pth <RVC目录>\assets\pretrained\f0G40k.pth --gguf work\f0G40k.train.gguf

# 3. 用 scripts/train_pipeline.ps1 或 rvc-train --train / --interactive 开始训练
```
