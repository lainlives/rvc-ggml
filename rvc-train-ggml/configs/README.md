# configs/ —— 训练配置模板

本目录放置训练配置模板。字段语义与原版 RVC 的官方配置一致（从官方仓库
`configs/v1/40k.json`、`configs/v2/48k.json` 原样拷贝），供
`rvc-train --train <config.json>` 与交互式向导（`rvc-train --interactive`）使用。

## 文件清单

| 文件 | 说明 |
|---|---|
| `v1-40k-f0.json` | v1 40k f0（segment 12800，hop 400，n_mel 125） |
| `v2-48k-f0.json` | v2 48k f0（segment 17280，hop 480，n_mel 128） |

## 配置结构

配置采用自有 schema（`train` / `data` / `model` 顶层段 + 可选 `run` 段）：

- `train` 段：`batch_size/epochs/learning_rate/betas/eps/lr_decay/c_mel/c_kl/log_interval/seed/segment_size`，
  以及本项目的扩展字段（可选）：`bucket/buckets/save_interval/disc_len/grad_accum`
  （`buckets` v1 仅取最大者作图 T）。
- `data` / `model` 段：与官方配置字段语义一致；`model.version`（v1/v2）、`model.has_f0`
  为本项目的显式声明字段，`d_periods` 未给出时按 version 推导。
- `run` 段（可选）：`exp/out/pretrain_g/pretrain_d/device/resume/export_every/snapshot_keep/
  export_run_json`。CLI 非空值优先于 `run` 段；带 `run` 段的配置可仅靠配置文件提供
  全部运行参数（`--train <config>` 不再强制 CLI 参数）。无 `run` 段的旧配置完全兼容。

## 默认训练参数

v1-40k-f0 / v2-48k-f0 默认：`batch_size=1, bucket=384, disc_len=12800`，
训练后端默认 `TRC_VK_CHUNK_NODES=64`。该组默认值按 4 GB 级显存/共享内存的 GPU 保守设定：

- 单个 `VkBuffer` / 存储范围仍受驱动硬限（启动日志会打印实际 `maxBufferSize` /
  `maxStorageBufferRange`，常见 4096 MiB）；`batch ≥ 2` 配大 `bucket/disc_len` 可能超限，默认不启用；
- train.cpp v0.4.0 起训练缓冲默认优先 device-local（独显真显存 + 共享 staging /
  可映射 device-local / host-visible 回退），不再固定占用系统内存；显存不足或旧驱动可用
  `TRC_VK_HOST_MEMORY=1` 回退 host-visible；
- 高显存机器可用 CLI `--batch/--bucket/--disc-len` 放大；
- 交互式向导另有更保守的稳定档默认（`bucket=300/batch=2/disc_len=4800`）。

路径类参数（数据、输出、权重等）不由配置文件提供，统一由 CLI 或向导给出。
