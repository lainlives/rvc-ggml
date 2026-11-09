# tools/ — Model conversion and validation tools (Python)

One-off conversion/validation tools, **not a runtime dependency of training** (training/preprocessing/export are all C++).

| Script | Purpose |
|---|---|
| `pth_to_train_gguf.py` | official RVC `.pth` (pretrained G/D, or the small inference model extracted by WebUI) → training-state GGUF (input format for finetuning/resume) |
| `verify_train_gguf.py` | training-state GGUF result validation: byte-for-byte comparison against the source `.pth` state_dict, or verification of the C++ load hash |

## Dependencies

- Python 3.9+ (only these two tools need `torch`; the training core itself needs no Python)

## Typical workflow (finetuning an official pretrained model)

```powershell
# 1. Official pretrained weights → training-state GGUF
python tools\pth_to_train_gguf.py --pth <RVC dir>\assets\pretrained\f0G40k.pth --config configs\v1-40k-f0.json --out work\f0G40k.train.gguf
python tools\pth_to_train_gguf.py --pth <RVC dir>\assets\pretrained\f0D40k.pth --config configs\v1-40k-f0.json --out work\f0D40k.train.gguf

# 2. (Optional) validate the conversion result
python tools\verify_train_gguf.py --pth <RVC dir>\assets\pretrained\f0G40k.pth --gguf work\f0G40k.train.gguf

# 3. Start training with scripts/train_pipeline.ps1 or rvc-train --train / --interactive
```
