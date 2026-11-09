# Numbers Handbook — Every Number in Training Parameters and Logs

> Audience: users. Explains what each **input parameter** (bucket/T, disc_len, batch, etc.) and each number in the **training log/CSV**
> means, what magnitude is normal, and what trends indicate. Read alongside the Usage Guide (`docs/usage_guide.md`); on errors, check its §8 FAQ first.

---

## 1. The Numbers in Input Parameters

### 1.1 Main Training Parameters

| Parameter | Unit/meaning | Typical value | Notes |
|---|---|---|---|
| `epochs` | epoch | 20–200 | One epoch = one full pass over all training samples |
| `batch` (B) | samples/step | 1–2 | Number of samples fed to the model per step. **Larger means more memory**: B≥2 with large T/d easily hits the Vulkan single-buffer/storage-range limit (commonly 4096 MiB) |
| `bucket` (T) | **frames** (1 frame ≈ 10 ms) | 200–384 | Upper bound on the time length of the training graph. **Only samples with frame count ≤ T are trained**; longer ones are skipped (counted in the log's `skipped`). A 3.7-second slice ≈ 368 frames, so `T=384` covers all official slices; `T=300` ≈ 3 seconds and skips longer slices. Larger T is slower and uses more memory |
| `disc_len` (d) | **samples** (at 40k sample rate, 1 second = 40000) | 1024–12800 | Length of the generated waveform the discriminator sees each time. `0` = use the whole segment_size (v1=12800, about 0.32 seconds). Larger d is slower and uses more memory, and has **the biggest impact on training speed** |
| `grad_accum` (K) | micro-batches | 1–8 | Gradient accumulation: accumulate K steps before one update, **effective batch = B × K**. The first choice when you want an equivalent large batch but lack VRAM |
| `steps` | optimization steps | — | Stop after N steps (for smoke tests/trial runs); takes precedence over `epochs` |
| `seed` | integer | 1234 | Random seed: determines sample shuffle order, etc. Same seed + same config → reproducible results |
| `learning_rate` | learning rate | 1e-4 | Official default; G/D share the starting point |
| `log_interval` | steps | 50 | How many optimization steps between step-log lines to the console/train.log |

**`segment_size` (no need to fill in; it lives in the config)**: the length of the internal segment randomly cut from a sample each training time (v1=12800 samples
≈ 0.32 s / 32 frames; v2=17280 / 36 frames). The difference from `bucket`: bucket decides "how long a sample the graph can hold",
segment_size decides "how long a piece of it is actually trained each time".

**Rough memory/speed rules** (measured; order-of-magnitude reference):

- VRAM/memory pressure ≈ determined jointly by `T`, `d`, and `B`; a single `VkBuffer`/storage range is still subject to
  the **Vulkan hard limit** (commonly 4096 MiB; see the startup log for the actual value; the program reports an explicit error).
  Since train.cpp v0.4.0, training buffers default to **device-local** (discrete-GPU real VRAM + shared staging; iGPU/ReBAR uses mappable
  device-local), no longer permanently occupying system memory; set `TRC_VK_HOST_MEMORY=1` to fall back to host-visible.
- Speed (seconds/step) depends mainly on `T × d × B`. Arc B570 measurements (v0.4.0, default device-local, 10-step A/B):
  `B=1/T=384/d=12800` ≈ **1.57 s/step**, `B=2/T=300/d=4800` ≈ **2.08 s/step**; with the
  `TRC_VK_HOST_MEMORY=1` fallback, about 25.6 / 39.4 s/step, with bit-identical loss trajectories. **If unsure, start from the wizard defaults
  (B=2/T=300/d=4800).**

### 1.2 Save/Export Parameters

| Parameter | Meaning |
|---|---|
| `save_interval N` | Save a resumable checkpoint every N epochs (`train.ep{N}.ckpt.gguf`) |
| `export_every N` | Export a **directly inferable** model every N epochs (`export.ep{N}.gguf`) — set it small (e.g. 5) if you want to "train a while, then listen" |
| `snapshot_keep N` | Number of snapshots to keep (0=keep all; 3 keeps only the latest 3, saving disk space) |

---

## 2. The Numbers in the Training Log

### 2.1 Startup Line (Most Important; Read This First)

```
[trainer] startup: bucket=384 batch=2 samples=92 skipped=0
          steps/epoch=46 epochs=50 max_steps=0 seed=1234 (est. total optimization steps ≈2300)
```

| Number | Meaning |
|---|---|
| `bucket / batch` | The T and B you supplied (echoed to confirm no typo) |
| `samples` | Number of **trainable** samples in the cache (those with frame count ≤ bucket) |
| `skipped` | Number of samples **skipped** because frames > bucket. **If this is not 0, T is too small** — some data did not take part in training (acceptable, but you should know) |
| `steps/epoch` | Optimization steps per epoch = floor(samples ÷ batch ÷ grad_accum) (the tail group that does not fill a batch is dropped) |
| `est. total optimization steps` | ≈ steps/epoch × epochs. Multiply it by your seconds/step = estimated total duration |

### 2.2 Step Line (One Line Every log_interval Steps)

```
[trainer] step 120 epoch 2 lr_g=9.975e-05 lr_d=9.975e-05
          loss_disc=1.834210 loss_gen_all=58.412211 loss_gen=1.204512
          loss_fm=9.871121 loss_mel=45.322104 loss_kl=2.014477
```

**Meaning of the 6 losses** (`loss_gen_all` = sum of the four on the right):

| Name | What it is | Who optimizes it |
|---|---|---|
| `loss_disc` | The discriminator's LSGAN loss: error in telling "real vs generated" audio | Discriminator D (lower = better separation) |
| `loss_gen_all` | **Generator total loss** = loss_gen + loss_fm + loss_mel + loss_kl | Generator G (overall metric) |
| `loss_gen` | The generator's adversarial loss: error in fooling the discriminator | G |
| `loss_fm` | Feature-matching loss: gap between generated and real audio on the discriminator's intermediate features | G |
| `loss_mel` | mel-spectrogram reconstruction error (L1) × **weight c_mel=45**. **Note the printed value already includes ×45**, so it is the largest and dominates gen_all | G (**the main engine of audio quality**) |
| `loss_kl` | KL divergence × weight c_kl=1: how close the posterior and prior distributions are | G |

**Other numbers**:

| Number | Meaning |
|---|---|
| `step` | Total optimization steps (one D update + one G update = 1 step; same convention as the original `global_step`) |
| `epoch` | Current epoch (1-based) |
| `lr_g / lr_d` | Current learning rate. Multiplied by `lr_decay` once per epoch (default 0.999875) — slowly shrinking during long training is normal |

### 2.3 Epoch Summary Line (One Line at the End of Each Epoch)

```
[trainer] epoch 2 summary step=92 lr_g=9.975e-05 loss_disc=... loss_gen_all=...
```

A snapshot of the last optimization step of the epoch + the learning rate after the epoch ends. `epochs.csv` records one row per epoch, making it easy to see trends between epochs.

### 2.4 End/Other Lines

| Log | Meaning |
|---|---|
| `buf_g=xxxx.x MiB buf_d=xxxx.x MiB` | Buffer size of the generator graph / discriminator graph. **A single one must not exceed the Vulkan single-buffer/storage-range hard limit (commonly 4096 MiB)**; exceeding it errors out immediately (rather than just being slow) |
| `step time: first=xxxx ms avg=xxxx ms` | **A slow first step is normal** (first allocation/warm-up); use the **average** to estimate total duration |
| `checkpoint saved: ... (step= epoch= lr_g=)` | Archive succeeded; the parenthesized values are the resume point |
| `epoch N exported inference GGUF: export.epN.gguf` | You can take this file to inference and listen to it |
| `skipped=N` (startup line) | See §2.1; a signal that T is too small |

### 2.5 Numbers in Files

| File | Contents |
|---|---|
| `train.log` | Human-readable full log (same as the console) |
| `loss.csv` | **One row per optimization step**: `step,epoch,lr_g,lr_d,loss_disc,loss_gen_all,loss_gen,loss_fm,loss_mel,loss_kl` — the easiest way to plot trends in Excel |
| `epochs.csv` | One row per epoch (same columns), for comparing epochs |
| `run.json` | Machine-readable summary of this run (when `--export-run-json` is enabled) |

---

## 3. Reading Trends: What Is Normal, What Warrants Caution

### 3.1 Normal Phenomena (No Need to Panic)

- **Large values on the first step and violent swings over the first few steps**: initial random state + an unformed discriminator; this is normal. Look at the trend after 10 steps.
- **`loss_disc` and `loss_gen` are non-monotonic and oscillate back and forth**: this is the essence of GANs — the two seek balance in a game,
  and **oscillating around some interval is healthy**. They are not "lower is always better" metrics.
- **`loss_mel` falls slowly and monotonically**: this is the **most reliable progress indicator** (the main engine of audio quality). Watch it on a scale of tens of steps.
- **`loss_kl` rises first, then stabilizes/slowly declines**: early in training the posterior is not yet aligned; this is a common trajectory.
- **`lr_g/lr_d` slowly shrink**: learning-rate decay is working.
- **The first step is very slow**: compilation/warm-up; look at the average.

### 3.2 Portrait of Healthy Training

- All 6 losses are **finite** (not `nan`/`inf`/huge numbers);
- `loss_mel` declines slowly overall, and `loss_gen_all` declines slowly along with it;
- `loss_disc`/`loss_gen` oscillate within some interval (e.g. each jumping around between 0.5 and 5);
- Step time is stable and does not get slower and slower over time.

### 3.3 Signals to Watch

| Signal | Possible cause | Action |
|---|---|---|
| `nan` / `inf` appears | Numerical divergence (F32 overflow) | Stop immediately; retry once with a lower learning rate; if it still reproduces, record the parameters (T/d/B) and logs and report them |
| `loss_gen` keeps rising + `loss_disc` approaches 0 | Discriminator too strong, suppressing the generator | Usually accompanied by audio-quality stagnation; you can reduce `disc_len` or listen early to the current export (many checkpoints remain usable) |
| `loss_mel` rises instead of falling and keeps worsening | Learning rate too large / abnormal config | Stop; retry with default hyperparameters |
| `loss_gen_all` permanently leaves the normal range after one spike | Training blew up (same handling as nan) | Resume from the most recent checkpoint or retrain |
| `skipped` share is high (e.g. >20%) | bucket too small; lots of data excluded | Increase bucket (mind memory), or accept training on shorter samples only |
| Step time gets slower and slower as training proceeds | Memory/system pressure | Reduce T/d/B; check background programs |

### 3.4 Magnitude Reference (Sanity Only)

Magnitudes vary widely with data and config; **watching trends matters more than absolute values**. Rough reference (v1-40k default weights, c_mel=45):

- `loss_mel` (already ×45): tens to a hundred early on, declining slowly with training;
- `loss_fm`: single digits to tens;
- `loss_gen` / `loss_disc`: oscillating within roughly 0.5–10 (adversarial balance);
- `loss_kl`: around single digits.

If your values stay more than an order of magnitude away from these ranges for a long time and do not converge, handle it per §3.3.

---

## 4. Common Conversions at a Glance

| Want to know | How to compute |
|---|---|
| How long is one frame | ≈10 ms (same for v1 and v2: hop = sample rate/100) |
| How long a sample can T=300 train | Slices within ≈3.0 seconds (longer ones are skipped) |
| How many samples one step consumes | batch × 1 (with grad_accum=K, an "effective large batch" update is synthesized every K steps) |
| Estimated total duration | Estimated total optimization steps × average step time (startup line + trailing step-time line) |
| How many steps per epoch | floor(samples ÷ batch ÷ grad_accum) (that is the startup line's steps/epoch) |
| Where training is now | step ÷ (steps/epoch) = current epoch; or read the epoch field of the step line |
| What to do when hitting the single-buffer/storage-range hard limit | Reduce any one of T, d, B; for an equivalent large batch use grad_accum |
