# RVC-GGML 技术文档（二）：ggml / Vulkan 实现细节

> 读者定位：了解 ggml 与 Vulkan 的读者。
> 本文重点讲**与原版不同的部分具体是怎么实现的**：模型布局、图构建技巧、量化方案、Vulkan 适配与坑。
> 只按需给出逻辑伪代码，不贴 C++ 源码。
>
> **重要**：本项目修改了本地 ggml 源码（Vulkan 矩阵乘 N 维切分，见 §5.5）。
> 补丁在 `patches/ggml-vulkan-matmul-n-split.patch`，升级 ggml 后必须重新应用。

---

## 一、模型格式与布局约定

### 1.1 基本约定

GGUF 中每个张量的 dims 顺序就是 ggml 的 `ne` 顺序（`ne[0]` 为 PyTorch 的最后一维）：

| PyTorch | ggml ne |
|---|---|
| Linear `[N, K]` | `[K, N]` |
| Conv1d `[O, I, K]` | `[K, I, O]` |
| ConvTranspose1d `[I, O, K]` | `[K, O, I]` |
| Conv2d `[O, I, KH, KW]` | `[KW, KH, I, O]` |
| ConvTranspose2d `[I, O, KH, KW]` | `[KW, KH, O, I]` |

导出时**不做任何转置**；`ggml_mul_mat(W, x)` 即 `Wᵀx`，恰为 PyTorch 的 `Linear(x)` 语义。
连续张量 `[C, T]` 在 ggml 中的内存布局与 PyTorch 行主序 `[T, C]` 相同，参考数据可直接 reshape 对照。

### 1.2 布局 v2：卷积权重展平（`rvc.conv_flat=true`）

**问题**：ggml 量化张量要求 `ne[0] % 32 == 0`（按行分块），而 Conv1d/Conv2d 权重的 `ne[0]` 是
kernel 宽（3/7/10/11/16…）→ 卷积权重无法量化。旧方案只能回退 F32（比 fp16 还大），
且 ggml 的 `ggml_conv_*` 组合把权重放在 `mul_mat` 的**第二操作数**，而 ggml 量化矩阵乘只支持
**第一操作数**是量化权重。

**方案**：导出时把卷积权重展平为"行宽 = 输入通道 × 核面积"的二维矩阵，运行时不再用 `ggml_conv_*`，
改为 `im2col + mul_mat(量化权重, cols)`：

```
Conv1d [O, I, K]      -> 存储为 (O, I*K)        # 行宽 I*K，满足 32 倍数则可量化
Conv2d [O, I, KH, KW] -> 存储为 (O, I*KH*KW)    # 行序 (i, kh, kw)，kw 最快 —— 与 im2col 列序一致
ConvTranspose1d/2d    -> 不展平（原生算子只支持 F32/F16 权重）
```

内存序的对应关系：PyTorch 行主序展平后天然满足 `i 慢、k 快`（Conv1d）与 `i 慢、(kh,kw) 快`（Conv2d），
与 ggml im2col 生成列向量时的索引顺序一致，因此展平操作不需要重排数据。

模型元数据增加布尔键 `rvc.conv_flat`，旧模型没有该键 → 走旧路径（自动兼容）。

---

## 二、图执行框架

### 2.1 GgmlGraphRunner

一个通用的"构建 → 分配 → 执行"容器：

- `ggml_init`（独立元数据上下文，默认 16 MiB 级）+ `ggml_new_graph_custom`
- 构建期 `expand(node)` 展开计算图
- `allocate()`：`ggml_gallocr_new/reserve/alloc_graph`，记录图缓冲大小与节点数
- `compute()`：单后端 `ggml_backend_graph_compute`（无调度器，因此**所有算子必须被目标后端支持**，
  否则直接报错——这也是为什么需要精确掌握 Vulkan 支持矩阵）
- 输入/输出通过 `ggml_backend_tensor_set/get` 读写；常量输入（相对位置嵌入、置换矩阵等）
  在图分配后上传一次

**图缓存策略**：
- HuBERT：按输入样本数缓存一张图；
- VITS：按 `frames/feature_dim/sid` 缓存；
- RMVPE：feature 图按帧数缓存；GRU/head 图固定 512 帧常驻；
- 管线长音频每块长度不同 → 每块重建（与 Python 一致），可用 `--x-*` 调整块长控制显存。

**实测缓冲规模**（5s 音频）：HuBERT ≈ 63 MiB；RMVPE feature ≈ 53 MiB（512 帧窗口）；
VITS ≈ 305 MiB（fp16，端到端管线的 698 帧输入）；量化档位因 im2col 列用 F32，VITS ≈ 492 MiB。

### 2.2 两种张量布局

| 区域 | 布局 | 原因 |
|---|---|---|
| enc_p / flow（线性、注意力、耦合层） | `[C, T]`（C 在 ne0） | 与 `mul_mat(W, x)` 的自然输入一致；相对位置/掩码都在 C 维切片 |
| dec（HiFiGAN）与音频前端卷积 | `[T, C]` | 与 `ggml_conv_*` 的自然输入一致，逐级上采样在 T 维 |
| 交换点 | enc_p 输出 → dec 输入、dec 输出 | 一次 `permute+cont` |

---

## 三、算子映射与实现技巧

### 3.1 线性层

```
linear(x, W, b): y = mul_mat(W, x)        # W 为 src0，可直接是量化张量
                 y = y + broadcast(b)      # bias 保持 F32/F16
```

### 3.2 卷积：原生路径与展平路径

```
conv1d_native(w3d, x):                    # 旧布局
    return ggml_conv_1d(w3d, x, stride, pad, dil)

conv1d_flat(w2d, x, stride, pad, dil):    # 新布局（rvc.conv_flat）
    I = x->ne[1]; O = w2d->ne[1]; K = w2d->ne[0] / I
    kernel = new_tensor_3d(F32, K, I, 1)   # 形状占位：im2col 只读形状，不读数据
    cols   = im2col(kernel, x, stride, pad, dil, dst_type = w2d 为 F16 ? F16 : F32)
    out    = mul_mat(w2d, reshape_2d(cols, K*I, OW*B))   # [O, OW*B]
    out    = permute_cont(reshape_3d(out, O, OW, B), 1,0,2,3)  # [OW, O, B]
    return out + bias
```

要点与坑：

1. **`ggml_im2col` 的 src0（kernel 参数张量）只用于推导形状**，CPU/Vulkan 实现都不读它的数据；
   因此用一个微型占位张量即可（标记 input，交给 gallocr 分配）。这也是展平方案能成立的关键。
2. **列向量类型**：权重是 F16 → 列用 F16（省内存）；权重是量化/F32 → 列必须用 **F32**，
   因为 ggml-cpu 在"src0 量化"路径要求 `src1` 是 F32（内部再转 Q8_0 系 vec_dot 类型）。
3. **输出需转回卷积顺序**：量化矩阵乘输出是 `[O, ...]`，而卷积生态期望 `[OW, O, B]`。
   注意 `ggml_permute(x, a0,a1,a2,a3)` 的参数语义是"源维 i 放到目标轴 aᵢ"：
   `[O,OW,B]→[OW,O,B]` 是 `(1,0,2,3)`；`[OC,OW,OH,N]→[OW,OH,OC,N]` 是 `(2,0,1,3)`。
4. **分组卷积**（HuBERT pos_conv，groups=16）：ggml 的 conv 不支持 groups。
   做法是按组把输入/权重切成 view（切完 `cont`，im2col 需连续内存），逐组 `ggml_conv_2d`（高度维=1 的退化），
   最后沿通道维 `concat`。该权重保持 F32/F16，不走量化。
5. **ConvTranspose1d**（生成器上采样，Vulkan 只支持 F32 权重）：`ggml_conv_transpose_1d(p0=0)` 输出两侧各多
   `pad=(k-s)/2`，用 `view_2d + cont` 裁掉。
6. **ConvTranspose2d**（RMVPE 解码器）：`ggml_conv_transpose_2d_p0` 输出 `2W+1`，
   与 PyTorch `(k=3,s=2,p=1,op=1)` 的关系是取 `full[1 : 1+2W]`，用 `view_4d` 偏移裁切。
7. **Conv2d 展平路径**：输出 `[OC, OW, OH, N]` 经 `permute_cont(2,0,1,3)` 得到 `[OW, OH, OC, N]`；
   batch-norm 的通道维是 ne[2]，与原生 conv 输出一致。

### 3.3 归一化与激活

- `LayerNorm`：`ggml_norm`（沿 ne0）→ `×γ + β`；
- `GroupNorm`：`ggml_group_norm` 的通道维是 ne[2]，HuBERT 前端逐通道归一化用 `reshape [T,1,C,1]` 实现；
- `BatchNorm2d`（RMVPE，eval 模式）：`(x - mean) / sqrt(var+eps) * γ + β`，参数 reshape 到 `[1,1,C,1]` 广播；
- GELU 用 erf 精确版 `ggml_gelu_erf`（fairseq `approximate='none'`）；
- 生成器内部 LeakyReLU 斜率 0.1，**最后一层是 0.01**（PyTorch 默认值），易错点。

### 3.4 注意力（HuBERT / enc_p）

多头注意力不是用 `ggml_flash_attn`，而是"逐头小矩阵乘"：

```
for h in heads:
    q_h,k_h,v_h = view2d(每个头, D 通道, T) + cont      # [D, T]
    score = mul_mat(k_h, q_h) * scale                   # [Tk, Tq]，softmax 沿 ne0=Tk
    out_h = mul_mat(transpose(v_h), softmax(score))     # [D, Tq]
concat(out_h)  # 头序与 hidden 索引一致
```

- HuBERT：绝对位置注意力（Post-Norm 结构，注意 residual 与 LayerNorm 的位置）；
- RVC enc_p：绝对 score 与"相对位置 logits"相加，后者通过以下两个变换对齐：
  - `relative_to_absolute`：相对 logits `[2T-1, T]` 经 pad/reshape/pad 得到 `[T, T]`（伪代码省略细节）；
  - `absolute_to_relative`：注意力权重的反向变换；
  - 相对位置嵌入由导出权重在主机侧展开为 `[D, 2T-1]` / `[2T-1, D]` 两个常量矩阵（区分转置序）。

### 3.5 Flow 的通道翻转

`flow` 中每层耦合前要做一次通道顺序翻转（奇数/偶数下标交换）。naive 做法是 192 次切片拼接；
这里用 **192×192 置换矩阵**乘以特征（`mul_mat`），一次矩阵乘完成，代价可忽略。

### 3.6 RMVPE 的静态图与状态接力

- **feature 子图**：DeepUnet 对 mel 做多层编码/解码。窗口推理时，图按
  `有效帧 4096 + 上下文 2×2048 = 8192 帧` 构建；**≤ 16352 帧（约 163.5 秒）**时整段一次推理，
  严格对齐 Python 的全量 mel 行为。
  该上限由 Vulkan 矩阵乘的 N 维约束反推（`N = 128×帧数 ≤ maxComputeWorkGroupCount[1]×32`，见 §5.5）；
  修复前设为 65536 帧，实际在 >163.8 秒时必然触发 `ggml-vulkan.cpp:8617` 断言，
  现已改为超限自动走窗口路径（窗口边界与 Python 有极小差异，听感一致）。
- **GRU 子图**：PyTorch 的 `nn.GRU`（双向）是逐时间步递归的，无法直接做成静态图。
  实现是**把 512 个时间步静态展开**，每步用上一隐状态算门控：

```
h = hidden0
for t in 0..511:
    gates = linear(x[t]; W_ih, b_ih) + linear(h; W_hh, b_hh)
    r, z, n = split(gates)                      # 门序与 PyTorch 一致
    n = tanh(n)                                 # 注意顺序：先算 n 再混合
    h = (1 - z) * n + z * h
    seq[t] = h
```

分块推理时"状态接力"：块首用参考实现同样的"keep 掩码"处理（首块清空、后续块携带上一块隐状态），
块尾把最终隐状态读出，作为下一块的初始隐状态。前后向分别处理，最后拼接送 360 类头。

### 3.7 检索（FAISS IVF 的等价语义）

```
选列表： 质心距离 L2²，取最近的 nprobe 个（nprobe 从索引元数据读取，必须与训练索引一致）
列表内： 向量 L2² 全量计算 → 全局 top-k（k=8）
加权：   w_i = (1/score_i)² / Σ(1/score_j)²
混合：   feats = index_rate * Σ w_i * vec_i + (1-index_rate) * feats
```

混合发生在 ×2 最近邻插值之前（与 Python 一致）。原索引由转换脚本导出为
`index.centroids / index.vectors / index.list_offsets / index.list_lengths` 四个 F32 张量 + 元数据。

---

## 四、量化实现（本项目的核心差异点）

### 4.1 ggml 量化的四条硬约束

1. 量化张量的 `ne[0]` 必须是块大小（32）的整数倍；
2. `ggml_mul_mat(A, B)` 的量化只允许出现在 **A（src0）**；
3. CPU 后端在 src0 量化时要求 src1 为 **F32**（内部转 vec_dot 类型）；
4. Vulkan 后端在 src0 非 F16/F32 且 src1 是量化类型时命中
   `GGML_ASSERT(y_non_contig || !qy_needs_dequant)`（即不支持"量化 B"），
   所以量化权重必须放 src0、激活列用 F32/F16。

### 4.2 展平卷积（让约束 1+2 同时满足）

算法见 3.2 的伪代码。配套的导出规则：

```
可量化判定：展平后行宽 % 32 == 0（且导出器本身能量化该值域）
敏感张量：bias / norm / embedding / GRU / 1D 参数等 -> F32
可量化但失败的张量 -> F16（不再回退 F32，避免"量化后反而更大"）
ConvTranspose 权重 -> 不展平，F32（Vulkan 原生算子要求）
```

### 4.3 混合量化策略（按模型）

| 模型 | 保持 F32 | 量化 |
|---|---|---|
| HuBERT | LayerNorm / final_proj / pos_conv / GroupNorm 参数 / 所有 1D 参数 | 前端卷积（展平）+ 注意力与 FFN 线性层 |
| RMVPE | BatchNorm/GRU/显著性头/1x1 shortcut/ConvTranspose/1D 参数 | DeepUnet 各级 3×3 卷积（展平；行宽不满足的如 16 通道层回退 F16） |
| 角色模型 | bias、`emb_g/emb_pitch/emb_phone`、`dec.m_source.l_linear`、`flow.*`、`dec.ups.*`、`dec.noise_convs.*` | enc_p 卷积（展平）、dec 残差块/前处理卷积（展平） |

各档位模型体积见《使用文档》。量化推理在 CPU/Vulkan 均通过验证（q8_0 端到端 corr ≈ 0.9995）。

---

## 五、Vulkan 适配与已知坑

### 5.1 设备要求

- 实例层：**Vulkan 1.2**（ggml 在 `vk::enumerateInstanceVersion() < 1.2` 时直接报错）；
- 设备层：必须支持 `storageBuffer16BitAccess`（Vulkan 1.1 核心特性或 `VK_KHR_16bit_storage` 扩展）；
- `VK_KHR_shader_float16_int8` 只在设备启用 fp16 计算路径时要求；本项目默认
  `GGML_VK_DISABLE_F16=1`，因此不强制该扩展（但官方新驱动基本都支持）。

### 5.2 两个默认规避项（程序启动时自动设置）

| 环境变量 | 默认 | 原因 |
|---|---|---|
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | 禁用整数点积 | Intel 驱动在量化模型多次图执行时会输出 NaN/全零（Intel 核显实测，ggml 0.18/0.23 均复现） |
| `GGML_VK_DISABLE_F16=1` | 禁用 fp16 计算路径 | Vulkan fp16 累加/反量化路径误差显著放大（实测音频 SNR 由 ~60 dB 掉到 ~33 dB），且核显上几乎无速度差异 |

可用 `RVC_VK_ENABLE_INTDOT=1` / `RVC_VK_ENABLE_F16=1` 恢复（不建议，除非确认驱动无问题）。

### 5.3 与本项目相关的 Vulkan 支持矩阵（ggml 0.23）

| 算子/路径 | Vulkan 支持 | 本项目用法 |
|---|---|---|
| `MUL_MAT`（src0 量化，src1 F32/F16） | ✅ 含 q4_0/q4_1/q5_1/q8_0；N 超大时由本项目补丁切分（§5.5） | 线性层、展平卷积 |
| `IM2COL`（src1 F32 → dst F16/F32） | ✅ | 卷积列提取（占位 kernel） |
| `CONV_TRANSPOSE_1D` | ⚠️ 仅 F32 权重/输出 | 生成器上采样（导出强制 F32） |
| `CONV_2D / CONV_TRANSPOSE_2D` | ⚠️ 权重 F32/F16、要求连续、`cin==b->ne[2]` | RMVPE（旧布局）与 ConvT2d |
| `GET_ROWS`（量化表） | ✅ | emb_pitch（本项目策略保持 F32，避免查表精度损失） |
| `CONCAT` | 要求两侧同类型/连续 | 分组卷积、skip 连接 |
| 其他（add/mul/norm/softmax/pool/permute/cont…） | ✅（F32/F16） | 图中大量使用 |

### 5.4 其他适配点

- **图构建顺序**：`concat`/`repeat` 等操作对连续性敏感，切片后先 `cont`；
- **大张量**：`ggml_mul_mat` 对超大 **A（M 维）**会按行分片（官方实现，`maxStorageBufferRange` 限制）；
  对超大 **B（N 维）**的切分为本项目对 ggml 的修改（见 §5.5）；
- **着色器编译**：首次运行每个算子配置需编译 SPIR-V 着色器，耗时数秒；驱动会缓存，
  后续启动明显加快；
- **内存**：权重常驻显存（hubert+rmvpe+角色 ≈ 440 MB fp16）；图缓冲按 2.1 的量级，
  长音频块越大越占显存。

### 5.5 本项目对 ggml 源码的修改：矩阵乘 N 维切分（重要）

**这是本项目相对官方 ggml 的唯一源码修改**，涉及文件
`src/ggml-vulkan/ggml-vulkan.cpp`，补丁见 `patches/ggml-vulkan-matmul-n-split.patch`。

**问题**：ggml 的 Vulkan 矩阵乘 `ggml_vk_matmul` 只对 batch/z 维切分，**N 维从不切分**。
dispatch 时 `wg1 = ceil(N / wg_denoms[1])`，须满足 `wg1 ≤ maxComputeWorkGroupCount[1]`（Vulkan 最低保证 65535）。
本项目相关路径（展平卷积）`wg_denoms[1] = 32`，因此 `N > 65535×32 = 2,097,120` 时触发：

```
ggml_vulkan.cpp: GGML_ASSERT(wg0 <= ... maxComputeWorkGroupCount[0] &&
                            wg1 <= ... maxComputeWorkGroupCount[1] &&
                            wg2 <= ... maxComputeWorkGroupCount[2]) failed
```

官方 ggml（含当前 master）只处理了超大 **M**（`needs_split`，`maxStorageBufferRange` 限制），没有 N 切分。

**触发场景**（修复前必崩）：

| 场景 | N 的构成 | 超限阈值 |
|---|---|---|
| VITS 生成器末级卷积（48 kHz，hop=480） | `VITS 帧数 × 480` | 单块 > 4369 帧（约 43.7 s） |
| 同上（40 kHz，hop=400） | `VITS 帧数 × 400` | 单块 > 5242 帧（约 52.4 s，接近分块上限，属侥幸未炸） |
| RMVPE 整段（非窗口）推理的展平卷积 | `128 × 帧数` | 帧数 > 16384（约 163.8 s） |

**修改内容**（逻辑，见补丁）：

```
ggml_vk_mul_mat(dst, src0, src1):
    max_n = maxComputeWorkGroupCount[1] × 32          # 32 = 所有 matmul pipeline 中最小的 wg_denoms[1]
    if 单批(ne2==ne3==1) and dst.N > max_n:
        for n_offset, cur_n in 按 max_n 列分块(dst.N):
            dst_view  = view(dst,  ne1=cur_n, offset += n_offset × nb1)
            src1_view = view(src1, ne1=cur_n, offset += n_offset × nb1)
            逐块矩阵乘(禁用 split_k)                    # 规避 split_k 归约网格 m×n 再次超限
        return
    ...原逻辑（超大 M 的切分抽成 ggml_vk_mul_mat_q_f16_m_split 供两处复用）...
```

要点：
1. 取 32（最小 `wg_denoms[1]`）做步长，保证无论后端选择哪个 pipeline（s/m/l、fp16/f32/量化）都不超限；
2. 分块内 `disable_split_k=true`：split_k 的归约 dispatch 元素数为 `m×n×batch`，同样受 workgroup 上限约束；
3. 只对 `ne2==ne3==1`（batch=1）生效，与官方 M 切分限制一致；其它情况走原路径；
4. 纯 dispatch 划分、不改数学，短音频/小模型完全不触发（行为与官方一致）。

**验证**（2026-09-17）：
- 强制切分（阈值临时降到 65535 列，让 5 s 样本也走切分）时，VITS v1/v2 自检 corr/SNR 与未切分**逐位一致**；
- 短样本回归：RMVPE 5 s、VITS v1/v2、e2e（Vulkan）全部通过；
- GPU 实测：48 kHz 模型 + 204 s 长音频完整转换通过（修复前第二块即断言）。

**重建须知**：
- 升级/替换 ggml 源码后必须重新应用补丁（FetchContent 路径下构建系统自动应用；
  本地 ggml 源码路径则由 CMake 检测补丁标记并 WARNING，需手动应用）；
- 未打补丁的官方 ggml 直接使用会在长音频上复现崩溃；
- 应用方法见 `patches/README.md`。

---

## 六、主机侧 DSP 与数值细节

| 模块 | 实现 | 与原版的差异 |
|---|---|---|
| 重采样 | 多相 FIR（有理比） | 与 librosa(soxr) 质量级一致，非逐样本 |
| 48 Hz 高通 | 手工 filtfilt（奇偶扩展 + 双次滤波） | 修复后与 scipy 逐位一致（此前的 padlen/扩展顺序 bug 已修） |
| mel 前端 | pocketfft 实 FFT + GGUF 中的滤波器组表 | 滤波器组与 librosa htk=True 一致 |
| 静音切块 | 能量曲线滑动窗口取最小 | 公式与 Python 相同 |
| SineGen | 主机侧逐样本生成正弦 + 固定种子噪声/注入噪声 | 相位与 f0 严格相关，f0 单帧差异会导致相位偏移 |
| RMS 混合 | 帧能量包络 | 与 Python 相同公式 |

---

## 七、验证与调试体系

- **参考数据**：Python 脚本在推理各中间层落盘（hubert/rmvpe/vits/e2e 四类）；
- **判定**：相关系数 corr + SNR（音频另看 max_err）；阶段阈值见技术文档一；
- **确定性**：验证模式注入参考噪声与参考 f0（`--noise-dir`、`--f0-inject`），隔离随机性与
  RMVPE 并列翻转；
- **调试环境变量**：`RVC_DEBUG_OP`（广播形状）、`RVC_DEBUG_CONV`（卷积形状/布局）、
  `RVC_DEBUG_LINEAR`、`RVC_HUBERT_DUMP`（中间张量落盘）、`RVC_RMVPE_DEBUG`（子图节点/缓冲）、
  `RVC_VK_ENABLE_INTDOT`、`RVC_VK_ENABLE_F16`。

---

## 八、性能与后续优化方向

- 参考量级（Intel 核显，5s 音频）：HuBERT ~0.2–1s，RMVPE ~1.2s（CPU）/~6.5s（Vulkan），
  VITS ~1.7–2.2s（Vulkan），全链路 Vulkan 约 4.5–6s（量化档位相近或略快）；
  入门级独显（Arc B570 级）预期显著快于实时。
- **待优化**：
  1. 量化卷积的 im2col 列在 Vulkan 上可改用 F16（省显存/带宽），CPU 仍用 F32；
  2. q4_0 混合策略按听感调参（生成器保持 F16 等）；
  3. 长音频显存：更细切块或跨块复用图缓冲；
  4. 非 ASCII 路径：命令行参数转 UTF-8 后再交给 ggml 文件接口。
