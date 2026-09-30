# RVC-GGML 对 ggml 源码的修改（本地补丁）

本项目使用的 ggml **不是官方原版**：为了让 **48 kHz 角色模型**与**超长音频**在 Vulkan 上不崩溃，
需要对官方 ggml 源码做一处必要修改。升级/更换 ggml 源码后**必须重新应用本补丁**，
否则长音频会再次触发 `maxComputeWorkGroupCount` 断言。

> 通过 FetchContent 自动获取 ggml 时（见根 `CMakeLists.txt` 的依赖解析），构建系统会
> 自动应用本补丁，无需手动操作；以下手动方法适用于使用本地 ggml 源码的场景。

## 补丁文件

- `ggml-vulkan-matmul-n-split.patch`
  - 目标文件：`src/ggml-vulkan/ggml-vulkan.cpp`（函数 `ggml_vk_mul_mat`，新增 `ggml_vk_mul_mat_q_f16_m_split`）
  - 基线：官方 ggml v0.23.0（该文件 20349 行；补丁后 20382 行，净增 33 行）

## 应用方法

在 ggml 源码树根目录执行（补丁路径按实际位置调整）：

```powershell
# 有 git 时
git apply -p1 <路径>\RVC-GGML\patches\ggml-vulkan-matmul-n-split.patch
# 或 GNU patch
patch -p1 < <路径>\RVC-GGML\patches\ggml-vulkan-matmul-n-split.patch
```

若 ggml 版本升级导致补丁无法直接应用，请按下方“修改说明”手动合并
（参考补丁中的上下文：把新增函数插到 `ggml_vk_mul_mat` 之前，并用新调用替换原 `needs_split` 分支里的 M 切分内联代码）。

## 背景

ggml 的 Vulkan 矩阵乘 `ggml_vk_matmul` 只对 batch/z 维切分，**N 维从不切分**；
shader 的 Y 方向 workgroup 数受 `maxComputeWorkGroupCount[1]`（Vulkan 最低保证 65535）限制。
矩阵乘按 `ceil(N / wg_denoms[1])` 分发（本项目相关路径 `wg_denoms[1] = 32`），因此当

```
N > 65535 × 32 = 2,097,120
```

时会在 `ggml_vulkan.cpp:8617` 触发：

```
GGML_ASSERT(wg0 <= ... maxComputeWorkGroupCount[0] &&
            wg1 <= ... maxComputeWorkGroupCount[1] &&
            wg2 <= ... maxComputeWorkGroupCount[2]) failed
```

官方 ggml（含当前 master）只处理了超大 **M**（`maxStorageBufferRange` 限制），没有 N 切分。

## 触发场景（修复前必崩）

| 场景 | N 的构成 | 超限阈值 |
|---|---|---|
| VITS 生成器末级卷积（48 kHz，hop=480） | `VITS 帧数 × 480` | 单块 > 4369 帧（约 43.7 s） |
| VITS 生成器末级卷积（40 kHz，hop=400） | `VITS 帧数 × 400` | 单块 > 5242 帧（约 52.4 s，接近分块上限，属侥幸未炸） |
| RMVPE 整段（非窗口）推理的展平卷积 | `128 × 帧数` | 帧数 > 16384（约 163.8 s） |

## 修改内容（`ggml_vk_mul_mat`）

1. 入口新增 N 切分：当 `dst->ne[2]==1 && dst->ne[3]==1 && dst->ne[1] > maxWGC[1]*32` 时，
   每 `maxWGC[1]*32` 列一块；用 `view_offs += n_offset * nb[1]` 构造 `dst` / `src1` 视图后逐块计算。
   （32 是所有 matmul pipeline 中最小的 `wg_denoms[1]`，取它可保证任意 pipeline 都不超限。）
2. 每块内禁用 split_k（`disable_split_k=true`），避免 split_k 归约网格 `m*n*batch` 再次超限。
3. 原有 M 切分内联代码抽成 `ggml_vk_mul_mat_q_f16_m_split`，N 切分块与普通路径共用。
4. 纯 dispatch 划分，不改变任何算子数学；仅在 N 超限时生效（短音频/小模型行为完全不变）。

## 验证（2026-09-17）

- **强制切分**（阈值临时降到 65535 列）下，VITS v1/v2 自检的 corr/SNR 与未切分**逐位一致**；RMVPE 302 s 通过
- 短样本回归：RMVPE 5 s、VITS v1/v2、e2e（Vulkan）全部通过
- GPU 实测：48 kHz 模型 + 204 s 长音频完整转换通过（修复前第二块即断言）

详细技术说明见 `docs\技术文档二_ggml_Vulkan实现细节.md` 第五节（5.5）。
