# configs/ — Training config templates

This directory holds training config templates. Field semantics match the original RVC's official
configs (copied verbatim from `configs/v1/40k.json` and `configs/v2/48k.json` in the official
repository), for use with `rvc-train --train <config.json>` and the interactive wizard (`rvc-train --interactive`).

## File list

| File | Description |
|---|---|
| `v1-40k-f0.json` | v1 40k f0 (segment 12800, hop 400, n_mel 125) |
| `v2-48k-f0.json` | v2 48k f0 (segment 17280, hop 480, n_mel 128) |

## Config structure

The config uses its own schema (`train` / `data` / `model` top-level sections + an optional `run` section):

- `train` section: `batch_size/epochs/learning_rate/betas/eps/lr_decay/c_mel/c_kl/log_interval/seed/segment_size`,
  plus this project's extension fields (optional): `bucket/buckets/save_interval/disc_len/grad_accum`
  (`buckets` v1 takes only the largest as graph T).
- `data` / `model` sections: field semantics match the official config; `model.version` (v1/v2) and `model.has_f0`
  are this project's explicit declaration fields, and `d_periods` is derived from the version when omitted.
- `run` section (optional): `exp/out/pretrain_g/pretrain_d/device/resume/export_every/snapshot_keep/
  export_run_json`. Non-empty CLI values take precedence over the `run` section; a config with a `run` section can supply
  all runtime parameters by itself (`--train <config>` no longer forces CLI parameters). Old configs without a `run` section remain fully compatible.

## Default training parameters

v1-40k-f0 / v2-48k-f0 defaults: `batch_size=1, bucket=384, disc_len=12800`,
and the training backend defaults to `TRC_VK_CHUNK_NODES=64`. These defaults are set conservatively for GPUs with 4 GB-class VRAM/shared memory:

- a single `VkBuffer` / storage range is still subject to the driver's hard limit (the startup log prints the actual `maxBufferSize` /
  `maxStorageBufferRange`, commonly 4096 MiB); `batch ≥ 2` with a large `bucket/disc_len` may exceed the limit, so it is off by default;
- since train.cpp v0.4.0, training buffers prefer device-local by default (real discrete-GPU VRAM + shared staging /
  mappable device-local / host-visible fallback), no longer permanently occupying system memory; with insufficient VRAM or an old driver, use
  `TRC_VK_HOST_MEMORY=1` to fall back to host-visible;
- machines with more VRAM can scale up via the CLI `--batch/--bucket/--disc-len`;
- the interactive wizard has a separate, more conservative stable-tier default (`bucket=300/batch=2/disc_len=4800`).

Path-like parameters (data, output, weights, etc.) are not provided by the config file; they are always supplied via the CLI or the wizard.
