# RVC-Train-GGML Usage Guide

> For users: the complete loop from **data preparation → training (finetune / from scratch) → export → inference**.
> Covers both **v1-40k-f0 and v2-48k-f0** model architectures.
> **Beginners should use the interactive wizard directly (just double-click the exe, see §0.6); the command-line/script approach is described in each section and is completely unaffected by the wizard.**
> **Training scale (epoch/step semantics, steps per epoch) is covered in §3.6 — required reading before a trial run.**
> **For the meaning of every number in the parameters and logs, and how to read trends, see the Numbers Handbook (`docs/numbers_handbook.md`).**

---

## 0. Overview and prerequisites

### 0.1 The three executables (must run as separate processes)

| Program | Responsibility |
|---|---|
| `rvc-prep.exe` | Audio preprocessing: slicing, 16k, RMVPE f0, HuBERT features, linear spectrogram, training cache, (optional) retrieval index |
| `rvc-train.exe` | Training (finetune / from scratch), logging, checkpoints, resume, export inference GGUF |
| `RVC-GGML.exe` | Inference (voice conversion; loads exported GGUF models) |

> **rvc-prep and rvc-train must be run as two separate processes** (two Vulkan backends cannot live in one process).
> All three exes are in `build\bin\` (or the release package root); relative paths in the commands below assume the repo/release package root as the working directory.

### 0.2 Runtime requirements and device conventions

- Platform: Windows 10/11 x64; the GPU must support **Vulkan** with working graphics drivers (NVIDIA/AMD/Intel all fine).
- **Zero Python and zero CUDA dependency at runtime**; `vulkan-1.dll` ships with the graphics driver.
- Device indices vary by machine: list them with `rvc-prep --list-devices` / `rvc-train --list-devices`, then pick one.
  - **Training**: pick the Vulkan GPU; CPU is for debugging only, **do not use it for real training** (very slow).
  - **Preprocessing**: auto-selects the GPU by default; `--device 99` = CPU (slow, debug only).
- For long training, set the environment variable `TRC_VK_CHUNK_NODES=64` first (mitigates the
  `VK_ERROR_DEVICE_LOST`/Windows TDR issues seen with some discrete GPU drivers):
  ```powershell
  $env:TRC_VK_CHUNK_NODES="64"
  ```
- Since train.cpp v0.4.0, training buffers prefer **device-local** by default (real VRAM on discrete GPUs + shared staging /
  mappable device-local / host-visible fallback); on old drivers or for troubleshooting you can set
  `TRC_VK_HOST_MEMORY=1` to fall back to host-visible.
- A single large buffer is subject to the **Vulkan single-buffer/storage-range hard limit** (driver-dependent; the actual value is printed in the startup log; commonly 4096 MiB):
  `batch ≥ 2` with a large `bucket/disc_len` may fail by exceeding the limit;
  to raise the effective batch use `--grad-accum K` (see §3.1/FAQ).

### 0.3 The role of Python

**There is zero Python dependency at runtime for training and preprocessing.** The only scenario requiring Python: converting the official RVC pretrained `.pth`
**one time** into training-state GGUF (§1.2; requires a Python environment with a working PyTorch). Training from scratch and inference need no Python at all.

### 0.4 Recommended entry point: the one-command driver script `scripts\train_pipeline.ps1`

If you want a one-command command-line flow, use this **generic driver script**: you decide the audio directory, cache directory, output directory, model and all parameters,
and it chains "preprocessing → training → export → (optional) index / closed-loop inference" and prints each command it actually executes.

**Starting from raw audio (automatic preprocessing + finetune + closed-loop inference)**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_pipeline.ps1 `
  -Audio D:\myvoice -Work work\myvoice `
  -Mode finetune -G work\models\f0G40k.train.gguf -D work\models\f0D40k.train.gguf `
  -Epochs 50 -Bucket 384 -DiscLen 12800 -ExportEvery 5 -Device 1 `
  -Infer -InferInput D:\some\test.wav -InferModelDir models
```

**Common variants**:

```powershell
# Train from scratch (random init, no G/D needed)
... -Audio D:\myvoice -Mode scratch -Epochs 200

# Existing cache, skip preprocessing
... -Cache work\myvoice -SkipPrep -Epochs 50

# Resume from checkpoint (requires matching .meta.json)
... -Cache work\myvoice -Resume work\train_myvoice\train.ckpt.gguf -Epochs 80

# Preprocessing only
... -Audio D:\myvoice -Work work\myvoice -PrepOnly

# Build retrieval index (for inference retrieval)
... -Cache work\myvoice -BuildIndex -IndexOut work\myvoice\myvoice.index.gguf

# Override hyperparameters (for advanced options not listed, put them in your own -Config)
... -Cache work\myvoice -Lr 1e-4 -Seed 1234 -Beta1 0.8 -Beta2 0.99 -CMel 45 -CKl 1

# Dry run first to see the commands (do not execute)
... -Cache work\myvoice -DryRun
```

Key parameters (full list in the script's `param` block):

| Parameter | Default | Description |
|---|---|---|
| `-Audio` | — | Raw wav directory (triggers `rvc-prep`) |
| `-Cache` | — | Existing cache directory (contains `dataset.json`; if given, preprocessing is skipped) |
| `-Work` | `work\rvc_project` | Directory where `-Audio` generates the cache |
| `-Out` | `work\train_<cache name>` | Training output directory |
| `-Config` | `configs\v1-40k-f0.json` | Base training config |
| `-Mode` | `finetune` | `finetune` (requires `-G/-D`) / `scratch` |
| `-G`/`-D` | — | Training-state weights (finetune) |
| `-Sr`/`-Version`/`-HasF0` | `40000`/`v1`/`1` | Preprocessing target sample rate/version/f0 |
| `-Hubert`/`-Rmvpe` | — | Preprocessing models (`hubert.gguf`/`rmvpe.gguf` paths) |
| `-PrepDevice` | `0` | rvc-prep device (0=Vulkan, 99=CPU) |
| `-Device` | `1` | rvc-train device (Vulkan GPU; 0=CPU debug) |
| `-Epochs/-Batch/-Bucket/-DiscLen/-GradAccum` | `20/1/384/12800/1` | Main training parameters |
| `-SaveInterval/-ExportEvery/-SnapshotKeep` | `1/1/0` | Snapshot/export/retention |
| `-Resume` | — | Resume checkpoint (requires `.meta.json`) |
| `-BuildIndex`/`-IndexOut` | — | Build retrieval index |
| `-Infer`/`-InferInput`/`-InferModelDir`/`-InferDevice`/`-Pitch`/`-Index`/`-IndexRate` | — | Closed-loop inference |
| `-DryRun`/`-PrepOnly`/`-SkipPrep`/`-Export` | — | Control switches |

Artifacts: `<Out>\` (`train.ckpt.gguf(+.meta.json)`, `train.ep{N}.ckpt.gguf`, `export.ep{N}.gguf`,
`loss.csv`/`epochs.csv`/`run.json`); with `-Infer` there is also `<Out>\converted.wav`.

### 0.5 Copying to another machine for training

A minimal set copy is enough (paths in the commands support **absolute paths**; directory relationships are not required):

- **Required**: `rvc-train.exe` (statically linked; at runtime it only depends on the system `vulkan-1.dll`, which ships with the graphics driver).
- **Additionally required for finetuning**:
  - Training-state weight GGUF (e.g. `f0G40k.train.gguf` + `f0D40k.train.gguf`, v1; for v2 use `f0G48k`/`f0D48k`);
  - The training cache directory (the whole directory produced by `rvc-prep`, containing `dataset.json`/`spec.json`/`samples\`);
  - The training config (`configs\v1-40k-f0.json` or `configs\v2-48k-f0.json`).
- Preprocessing can be done on any machine; if the target machine only trains, copy `rvc-train.exe` + weights + cache + config.
- Verify the environment on the target machine first: `rvc-train.exe --selftest` (prints devices and a compute self-test);
  for long training set `TRC_VK_CHUNK_NODES=64` (§0.2).
- For small-scale trial training parameters see §3.1 and §3.6.

### 0.6 Interactive usage (recommended for beginners: double-click and go)

Both programs have a built-in **wizard**: **double-clicking the exe (or running it with no arguments in a terminal) enters the wizard**, where you answer each question and work starts.
Before exiting, the program displays "Press Enter to exit" and does not flash-close (when launched from cmd/PowerShell/a script it does not pause, so it does not block batch processing).
**The command-line approach is completely unchanged** (the commands in each section still work). The two programs must **run separately** (§0.1); the full flow has three steps:

**Step 1 — Preprocessing (double-click `rvc-prep.exe`)**:

| Wizard question | How to answer |
|---|---|
| Input audio directory (or a single wav file) | Your raw wav directory (§2.1) |
| Preprocessing target (1=v1-40k / 2=v2-48k) | Must match the model version used for training |
| RMVPE weights path | `rmvpe.gguf` under the RVC-GGML model directory (press Enter to skip f0; **recommended for training**) |
| HuBERT weights path | `hubert.gguf` (**required for training**; leaving it empty warns "cache cannot be used for training") |
| Training cache output directory | Defaults to `work\cache_auto_40k` / `cache_auto_48k` |
| Compute device index | Enter = auto (GPU); `99` = CPU (slow, debug only) |

After confirmation preprocessing runs; when it finishes it **automatically self-validates**, printing the sample count and the "next step: run rvc-train" hint.

**Step 2 — Training (double-click `rvc-train.exe`)**:

| Wizard question | How to answer |
|---|---|
| Data entry (1=existing cache / 2=raw audio directory) | 1 = enter the cache directory from the previous step; 2 = enter the raw audio directory and the wizard will **invoke `rvc-prep.exe` from the same directory to preprocess for you**, then continue |
| Model source (1=finetune / 2=from scratch) | Finetune requires G/D training-state GGUF (§1.2); from scratch needs no weights |
| Output directory | Log/checkpoint/export directory; **if the directory already has a checkpoint it asks whether to resume** (requirements in §3.4) |
| epochs / batch_size / bucket / disc_len | Default = **safe profile `batch=2 / bucket=300 / disc_len=4800`**; bucket>384 or disc_len>6400 triggers the **large-config cost warning** (large configs cost more time per step/VRAM; start with a small trial run, see §3.6) |
| learning_rate / seed | Default `1e-4` / `1234` |
| Training device index | The wizard **lists the devices** (name/type/VRAM) and then asks for a number; `-1`=auto |
| Configure advanced parameters? (y/N) | Enter to skip; `y` lets you set grad_accum (gradient accumulation) / save_interval / export_every / snapshot_keep / log_interval |

Before confirmation the program prints the **cache info** (sample count/version/sample rate) and the **estimated optimization steps/epoch**; a mismatch between the cache version and the model config triggers a
**warning** (training will be rejected by the preflight checks). After confirmation it writes `<out>\train_config.json` and starts training immediately.

**Step 3 — Inference**: use the `export.ep{N}.gguf` from §4 and run inference in `RVC-GGML.exe` with the commands in §5.

**Scripted wizard** (non-interactive batch): `rvc-prep --answers <file>`, `rvc-train --interactive --answers <file>`
(one answer per line, in the same order as the questions).

---

## 1. One-time asset preparation

### 1.1 Models needed for preprocessing (shipped with / converted for RVC-GGML)

- `hubert.gguf` (HuBERT features), `rmvpe.gguf` (f0): put them in RVC-GGML's `models\` directory
  (or any directory you choose; just fill in the path in the wizard/command line).
- Inference character models (e.g. `xxx.gguf`) and retrieval indexes (`*.index.gguf`): same as above, inference-only; training does not need them.

### 1.2 Finetune starting point: official pretrained `.pth → training-state GGUF`

Finetuning starts from the official RVC pretrained weights: v1 uses `f0G40k.pth` (G) + `f0D40k.pth` (D);
v2 uses `f0G48k.pth`/`f0D48k.pth` under `pretrained_v2`.

Use the one-time converter (requires a Python environment that can run PyTorch; training/inference do not need Python at runtime) to convert them into **training-state GGUF**:

```powershell
python tools\pth_to_train_gguf.py --pth <official pretrained>\f0G40k.pth `
    --config configs\v1-40k-f0.json --out work\models\f0G40k.train.gguf
python tools\pth_to_train_gguf.py --pth <official pretrained>\f0D40k.pth `
    --config configs\v1-40k-f0.json --out work\models\f0D40k.train.gguf
# Independent verification (optional; per-tensor byte-level + CRC)
python tools\verify_train_gguf.py --pth <official pretrained>\f0G40k.pth `
    --gguf work\models\f0G40k.train.gguf --hashes work\models\f0G40k.hashes.txt
```

- For v2, replace `--config` with `configs\v2-48k-f0.json` and the weights with f0G48k/f0D48k.
- `--kind auto|g|d` (default auto-infer), `--dtype f32|f16` (default f32), `--version v1|v2`, `--f0 0|1` as needed.
- Inference small models (without `enc_q`) are not used for training.

---

## 2. Data preparation (`rvc-prep`)

### 2.1 Collecting and placing audio

- Goal: **10–50 minutes** of clean, single-speaker, low-reverb vocals (training quality depends mainly on the data).
- Sample rate: same as the target model (v1-40k → `--sr 40000`; v2-48k → `--sr 48000`).
- Put them in one directory (supports `.wav`; a single wav also works), e.g. `work\myvoice\`.

### 2.2 Generating the training cache

**Recommended**: just double-click `rvc-prep.exe` and use the wizard (see §0.6 step 1); command-line approach:

```powershell
build\bin\rvc-prep.exe --input work\myvoice --sr 40000 --version v1 --has-f0 1 `
    --exp work\cache_myvoice `
    --rmvpe models\rmvpe.gguf `
    --hubert models\hubert.gguf `
    --device 0
```

- `--input <dir|wav>`: audio directory or a single wav.
- `--sr`: target sample rate (default 40000); `--version v1|v2` determines the HuBERT feature dimension (v1=256 / v2=768).
- `--has-f0 1`: f0 model (the target of this project); `--no-f0` / `--no-feat` / `--no-spec` disable the corresponding outputs (generally not used).
- `--per 3.7 --overlap 0.3`: official chunking parameters (defaults).
- `--device 0` = Vulkan (default); `--device 99` = CPU (very slow, debug only).
- `--debug-dump <dir>`: optional, dumps raw intermediate data (for troubleshooting).

Cache outputs (`work\cache_myvoice\`):

```
dataset.json / spec.json
samples\<id>.gt.bin / .16k.bin / .f0hz.bin / .f0bin.bin / .feat.bin / .spec.bin
```

### 2.3 Validating the cache

```powershell
build\bin\rvc-prep.exe --validate-cache work\cache_myvoice
```

### 2.4 (Optional) Building the retrieval index (inference retrieval only; training does not need it)

RVC retrieval (`index_rate > 0`) is an **inference-time** feature. If you want retrieval mixing at inference time, build `index.gguf` from the cache features:

```powershell
build\bin\rvc-prep.exe --index --exp work\cache_myvoice --out work\index_myvoice\myvoice.index.gguf `
    --version v1 --seed 1234
build\bin\rvc-prep.exe --validate-index work\index_myvoice\myvoice.index.gguf
```

- Full IVF by default; `--index-mvp` falls back to a single list; `--nprobe N` writes the number of retrieval lists (default 1, adjustable per RVC-GGML needs);
  `--seed` guarantees reproducibility.

---

## 3. Training (`rvc-train`)

The config uses its own schema: three top-level sections `train` / `data` / `model` (the repo ships `configs/v1-40k-f0.json`,
`configs/v2-48k-f0.json`), plus an optional `run` section. `rvc-train --train <config.json>` can rely solely on the `run` section for
`exp/out/pretrain_g/pretrain_d/device`, and the CLI can override them as well.

> Device indices vary by machine: run `rvc-train --list-devices` first; `--device 1` in the examples below means
> "the Vulkan GPU in the list"; the CPU index (usually 0) is for debugging only.

### 3.1 Finetuning (recommended starting path)

```powershell
build\bin\rvc-train.exe --train configs\v1-40k-f0.json `
    --exp work\cache_myvoice `
    --weights work\models\f0G40k.train.gguf `
    --d-weights work\models\f0D40k.train.gguf `
    --out work\train_myvoice `
    --device 1 `
    --epochs 20 --batch 1 --bucket 384 --disc-len 12800 `
    --save-interval 1 --export-every 5
```

- `--weights`/`--d-weights`: finetune starting point (= `--pretrain-g`/`--pretrain-d`, aliases).
- `--epochs`: number of training epochs; you can also use `--steps N` to cap the total number of optimization steps (smoke/test).
- `--bucket`: graph T (must be ≥ `segment_frames`; samples exceeding the bucket are skipped and counted in the log).
- `--disc-len`: discriminator input length (0 = official full `segment_size`; for tests you can set a small value such as 1024 for speed).
- `--grad-accum K`: gradient accumulation (effective batch = batch × K); **the preferred way to raise the effective batch** (B≥2 is limited by the Vulkan single-buffer/storage-range hard limit, commonly 4096 MiB).
- `--save-interval N`: save `train.ep{N}.ckpt.gguf` every N epochs (resumable).
- `--export-every N`: export inference GGUF `export.ep{N}.gguf` every N epochs.
- `--snapshot-keep N`: number of epoch snapshots to keep (0=keep all).
- `--export-run-json <path>`: write a run summary JSON (step time/memory/first-steps loss).

### 3.2 Training from scratch

```powershell
build\bin\rvc-train.exe --train configs\v1-40k-f0.json `
    --exp work\cache_myvoice --out work\train_scratch_myvoice `
    --scratch --device 1 --epochs 200 --batch 1 --bucket 384 --disc-len 12800 `
    --save-interval 1 --export-every 10
```

- `--scratch`: generate randomly initialized G/D per the config's `data/model` sections (specs aligned with official v1-40k); no pretrained weights needed.
- Mutually exclusive with `--weights/--pretrain-g`. **From-scratch quality depends on data volume**; finetuning is usually the primary path.

### 3.3 Interactive training wizard

**Double-clicking `rvc-train.exe` (or running it with no arguments) enters it**; you can also use `--interactive` explicitly:

```powershell
build\bin\rvc-train.exe --interactive
# or supply a base config / answers file (for scripting)
build\bin\rvc-train.exe --interactive --config configs\v1-40k-f0.json --answers work\answers.txt
```

The wizard asks in order (press Enter to accept the `[bracketed]` default; see §0.6 step 2 for details):

1. **Data entry**: 1=existing cache directory / 2=raw audio directory (the wizard invokes the same-directory `rvc-prep.exe` and continues after preprocessing;
   if the exe is not found it prints the equivalent command line).
2. **Model source**: 1=pretrained finetune (enter G/D training-state GGUF) / 2=train from scratch.
3. (finetune) **G / D paths** → **output directory** (if a checkpoint exists, asks **whether to resume**).
4. **epochs / batch_size / bucket / disc_len** — without a base config the default is the **stable profile
   `batch=2 / bucket=300 / disc_len=4800`** (values from the CLI/base config take precedence); large configs trigger the cost warning.
5. **learning_rate / seed / device** (devices are listed first, then you pick a number; `-1`=auto).
6. **Advanced parameters** (y/N, collapsed by default): grad_accum / save_interval / export_every / snapshot_keep / log_interval.
7. **Pre-confirmation summary**: cache sample count/version/sample rate + estimated optimization steps/epoch + version mismatch warning + the full config.
   After confirmation it writes `<out>\train_config.json` and **starts training immediately**.

### 3.4 Resuming from a checkpoint

```powershell
build\bin\rvc-train.exe --train configs\v1-40k-f0.json `
    --exp work\cache_myvoice `
    --weights work\models\f0G40k.train.gguf --d-weights work\models\f0D40k.train.gguf `
    --out work\train_myvoice `
    --resume work\train_myvoice\train.ckpt.gguf `
    --device 1 --epochs 40
```

**Hard requirements for resuming**:

- The checkpoint must exist together with the **same-name sidecar metadata file** `<ckpt>.meta.json` (written automatically by training). Missing meta → **error and refusal**.
- The resume config must be **consistent** with the saved one: `batch_size / bucket / grad_accum / seed / segment_size / disc_len / lr /
  lr_decay / c_mel / c_kl / model structure / data dims`, etc., plus the **sample set** (sample count + total frames + id list).
  Any mismatch produces an explicit error; **no silent misalignment**. If you want to change the config, start training over.
- You can press `Ctrl+C` at any time during training and resume from the rolling checkpoint (`train.ckpt.gguf`);
  you can also resume from the epoch snapshots written by `--save-interval`.

### 3.5 Training artifacts (`--out` directory)

```
train.log / loss.csv / epochs.csv        # per-step/per-epoch logs
train.ckpt.gguf + train.ckpt.meta.json   # final rolling checkpoint (includes resume metadata)
train.ep{N}.ckpt.gguf + .meta.json       # snapshot every N epochs (--save-interval)
export.ep{N}.gguf                        # inference GGUF exported every N epochs (--export-every)
train.ep{N}.snapshot.gguf                # exported intermediate state (normally no manual use)
train_config.json                        # written only in wizard mode
```

### 3.6 Training scale: relationship between `epoch` / `step` and original RVC (required reading before a trial run)

**In one sentence**: this project is **isomorphic** to the original — **one epoch = one original epoch = one full pass over the training samples; one step = one original
`global_step` = one optimizer update (one D update + one G update)**. `epoch = step / steps_per_epoch`; crossing an integer boundary enters
the next epoch (at the same time the G/D learning rate decays once per epoch, matching the original ExponentialLR).

**Steps per epoch**:

```
steps_per_epoch = floor( trainable sample count / batch_size / grad_accum )      # a trailing partial group is dropped
```

- **Trainable sample count** = samples in the cache satisfying `segment_frames ≤ sample frames ≤ bucket`; the rest are counted as `skipped` in the startup log.
  The default `bucket=384` covers all 3.7s slices (`skipped=0`).
- **How to check**: at **startup** the console prints one line (also written to `train.log`), which you can read directly:
  ```
  [trainer] startup: bucket=384 batch=1 samples=92 skipped=0 steps/epoch=92 epochs=20
            max_steps=0 seed=1234 (estimated total optimization steps ~= 1840)
  ```

**Two differences from the original (these determine what values to use in your tests)**:

1. **batch_size**: the original `config.json` defaults to **4**, this project defaults to **1**. One original step consumes 4 samples; one step here consumes 1.
   To "match one original step in data volume", use `--batch 4` (limited by the Vulkan single-buffer/storage-range hard limit, commonly 4096 MiB; B≥2 is often insufficient) or
   **`--grad-accum 4`** (accumulate 4 micro-batches and then update once; VRAM-friendly, approximately equivalent gradients).
2. **Same sample-count convention**: original sample count = number of slices in `0_gt_wavs`; this project's sample count = cache sample count (**the same slicing**,
   so the counts are basically identical). Therefore `steps_per_epoch` matches the original for the same batch size.

**Comparison example**: original `batch_size=4`, slices N=1000 ⇒ original `steps/epoch=250`; this project with `--batch 4` also gives `250`
(one epoch to one epoch); if this project uses the default `--batch 1` then `steps/epoch=1000` (small steps, faster iteration, lower VRAM, but each step has 1/4 the data volume of the original).

**`--steps N`**: run only N optimization steps (counted across epochs), for smoke tests; takes precedence over `--epochs`.

**Practical constraints of scale vs hardware (measured)**: large `bucket`/`disc_len` configs have noticeably higher step time and VRAM cost.
Since train.cpp v0.4.0 training buffers prefer device-local; on an Arc B570 a 10-step A/B measurement gave about **1.57 s/step** for
`B=1/T=384/d=12800` and about **2.08 s/step** for `B=2/T=300/d=4800`
(about 25.6 / 39.4 s/step when falling back to host-visible). Even so, before a long production run it is still recommended to trial-train with a small config,
confirm stability and then scale up; the wizard defaults are the safe profile, and entering large values triggers a warning.

**Small-scale trial recommendations**: first use `--bucket 200 --disc-len 1024` to shrink the graph and discriminator input (faster, lower memory),
`--epochs 1~3` or `--steps 5~20`, confirm the loss is finite and trends sensibly and that checkpoints can be saved, then scale up to production parameters.

---

## 4. Exporting the inference GGUF

During training `--export-every N` exports `export.ep{N}.gguf` in real time; you can also export offline from **any training-state GGUF**:

```powershell
build\bin\rvc-train.exe --export-gguf work\train_myvoice\train.ep5.snapshot.gguf `
    --out work\train_myvoice\myvoice.export.gguf
```

Exported content: folds `weight_norm` (`w = g·v/‖v‖`), flattens Conv1d weights (except `dec.ups.*`), exports only the inference subset
(`enc_p / flow / dec / emb_g`), writes `rvc.*` metadata; the artifact can be **loaded directly by RVC-GGML**.

---

## 5. Inference (RVC-GGML)

Convert once with the exported model:

```powershell
build\bin\RVC-GGML.exe `
    --input <input wav> `
    --model work\train_myvoice\myvoice.export.gguf `
    --model-dir models `
    --output work\train_myvoice\out.wav `
    --rms-mix-rate 0.75 --pitch 0 --device 0
# Enable retrieval (when an index is available):
#   --index work\index_myvoice\myvoice.index.gguf --index-rate 0.75
```

- `--model-dir`: directory containing the hubert/rmvpe models (default `models`).
- `--device 0` = Vulkan, `--device 99` = CPU (indices per `--list-devices`/startup output).
- You can also double-click `RVC-GGML.exe` (no arguments) to enter its own interactive wizard.

---

## 6. Complete example (from your audio to inference)

The following demonstrates the complete loop with generic paths (`<wav directory>` = your raw audio, `models` = directory containing hubert/rmvpe):

```powershell
# 0) (Optional) long-training environment variable
$env:TRC_VK_CHUNK_NODES="64"

# 1) Preprocessing (v1-40k; for v2 switch to --sr 48000 --version v2)
build\bin\rvc-prep.exe --input D:\myvoice --sr 40000 --version v1 `
    --exp work\cache_myvoice `
    --rmvpe models\rmvpe.gguf --hubert models\hubert.gguf --device 0

# 2) Validate the cache
build\bin\rvc-prep.exe --validate-cache work\cache_myvoice

# 3) Finetune for 20 epochs, checkpoint and export an inference model every 5 epochs (fill the device index per --list-devices)
build\bin\rvc-train.exe --train configs\v1-40k-f0.json `
    --exp work\cache_myvoice `
    --weights work\models\f0G40k.train.gguf --d-weights work\models\f0D40k.train.gguf `
    --out work\train_myvoice --device 1 `
    --epochs 20 --batch 1 --bucket 384 --disc-len 12800 `
    --save-interval 1 --export-every 5

# 4) Convert an audio clip with the exported inference model
build\bin\RVC-GGML.exe `
    --input D:\some\test.wav `
    --model work\train_myvoice\export.ep20.gguf `
    --model-dir models `
    --output work\train_myvoice\out.wav --rms-mix-rate 0.75 --device 0
```

Equivalently, **steps 1 and 3 can both be done by double-clicking the corresponding exe and using the wizard** (§0.6).

---

## 7. Config field quick reference (own schema)

Three top-level sections + optional `run`. Taking `configs/v1-40k-f0.json` as an example:

| Section | Key fields | Description |
|---|---|---|
| `train` | `batch_size/epochs/learning_rate/betas/eps/weight_decay/lr_decay/c_mel/c_kl` | Optimization and loss (AdamW + ExponentialLR) |
| `train` | `bucket/disc_len/segment_size/save_interval/seed/grad_accum/log_interval` | Data/training control |
| `data` | `sampling_rate/filter_length/hop_length/win_length/n_mel_channels/mel_fmin/mel_fmax` | Audio and spectrogram (must match the cache) |
| `model` | `version/has_f0/inter_channels/hidden_channels/.../upsample_rates/d_periods` | Model structure (from-scratch training generates from this; finetune must match the weights) |
| `run` (optional) | `exp/out/pretrain_g/pretrain_d/device/resume/export_every/snapshot_keep/scratch/export_run_json` | Run wiring; CLI overrides take precedence |

- **Not implemented**: `init_lr_ratio/warmup_epochs` (kept in the config but ignored), `fp16_run` ignored (training is fixed to F32).
- Changing fields in `train/data/model` that affect the training trajectory will cause resume to be rejected (see §3.4).

---

## 8. FAQ

- **Can training use the CPU?** No, not as a training target. The CPU backend is very slow and debug-only; use a Vulkan GPU for real training.
- **Training is slow / it is using shared memory?** Since train.cpp v0.4.0 device-local is preferred by default: discrete GPUs use real VRAM
  (through shared staging when unmappable), integrated GPUs/ReBAR use mappable device-local; you can still use
  `TRC_VK_HOST_MEMORY=1` to fall back to the old host-visible behavior. To speed up, first reduce
  `bucket`/`disc_len`/`batch`, or raise the effective batch with `--grad-accum`.
- **`VK_ERROR_DEVICE_LOST` (long training on a discrete GPU)?** Set `TRC_VK_CHUNK_NODES=64` (batched submission); if it still reproduces, record
  the machine model/driver and the `TRC_VK_CHECK=1` output as feedback.
- **`batch>1` fails / exceeds the limit?** Vulkan single-buffer/storage-range hard limit (commonly 4096 MiB; see the startup log for the actual value);
  raise the effective batch with `--grad-accum`, or reduce
  `bucket`/`disc_len`.
- **Resume reports "missing meta / inconsistent config"?** See §3.4; old checkpoints without `.meta.json` cannot be resumed.
- **Cache and config mismatch (`cache spec.hop/target_sr/n_fft mismatch`)?** `--sr/--version` must match those used when the cache was generated.
- **Reports `model and training cache mismatch: G weights are v1 (feature dim 256), but cache feature dim is 768…`?** = model version and cache version
  mismatch. Align as follows: **v1** uses f0G40k/f0D40k + v1 cache + `configs\v1-40k-f0.json`; **v2** uses f0G48k/f0D48k +
  v2 cache + `configs\v2-48k-f0.json`. G/D must also both be v1 or both v2.
- **hubert/rmvpe not found (inference)?** Point `--model-dir` at the directory containing `hubert.gguf`/`rmvpe.gguf` (default `models`).
- **How to verify the installation/environment?** `rvc-prep --list-devices`, `rvc-train --list-devices`, `rvc-train --selftest`,
  `rvc-prep --selftest`; or just double-click either exe to enter the wizard.
- **Garbled console text?** The program automatically switches the console to UTF-8 at startup. If it is still garbled (some old PowerShell 5.1 builds),
  run `chcp 65001` first; Windows Terminal / PowerShell 7 is easier.
- **Diagnostics/self-test switches (not for daily use)**: `rvc-train --selftest-forward/-backward/-loss/-train-step/-train-loop/-data`,
  `--selftest-trainer/-grad-accum/-resume/-wizard/-scratch`; for script cross-checks see `tools\README.md`.
