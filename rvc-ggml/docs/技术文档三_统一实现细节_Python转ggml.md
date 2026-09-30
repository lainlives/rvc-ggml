# RVC 推理：从 Python/PyTorch 到 C++/ggml 的统一实现细节

> 本文是自包含的技术文档：只描述实现机制、算子映射、流水线与后端约束。
> 阅读者无需接触参考实现或原始工程，仅凭本文即可复现整体流程与关键算法。
> 内容以伪代码与数学定义为主，不涉及工程进度、发布与验证记录。

---

## 一、总体数据流

一次转换由一条固定流水线完成，全部为静态计算图 + 主机侧 DSP：

```
WAV (任意采样率/声道)
  │ 下混 + 多相 FIR 重采样
  ▼
单声道 16 kHz
  │ 48 Hz 零相位高通
  ▼
整段 f0 估计（RMVPE，基于反射补齐后的整段音频）
  │
  ├─ 按低能量点切块（x_pad / x_query / x_center / x_max，单位：秒）
  │
  ▼  逐块：
  HuBERT 特征（v1: 第 9 层 + final_proj→256；v2: 第 12 层→768）
  →（可选）FAISS IVF 检索特征混合
  → ×2 最近邻上采样
  → protect（清音帧用原始特征按比例回填）
  → 变调后的 pitch bin + RMVPE f0
  → SineGen 激励（f0 模型）→ tanh(l_linear(·))
  → VITS（enc_p → flow 逆 → HiFiGAN NSF 解码）
  → 按 x_pad（目标采样率）裁掉首尾
  │
  ▼ 拼接所有块
RMS 响度混合 →（可选重采样）→ 峰值归一化 → int16 WAV
```

数据空间约定：

| 阶段 | 采样/帧率 | 维度记号 |
|---|---|---|
| 波形（模型域） | 40 kHz / 48 kHz | 1×N |
| 波形（特征域） | 16 kHz | 1×N |
| HuBERT 特征 | 50 fps | `[T, C]`，C=256/768 |
| VITS 帧 | 100 fps | `[T, 192]` 语义（含 ×2 后） |
| pitch bin / f0 | 100 fps | `[T]` int32 / `[T]` float |

---

## 二、模型表示（GGUF）

### 2.1 维度约定

GGUF 以 ggml 的 `ne` 顺序存储维度，`ne[0]` 等于 PyTorch 形状的最后一维。导出时不做任何转置：

| PyTorch 形状 | ggml `ne` |
|---|---|
| Linear `[N, K]` | `[K, N]` |
| Conv1d `[O, I, K]` | `[K, I, O]` |
| ConvTranspose1d `[I, O, K]` | `[K, O, I]` |
| Conv2d `[O, I, KH, KW]` | `[KW, KH, I, O]` |
| ConvTranspose2d `[I, O, KH, KW]` | `[KW, KH, O, I]` |

由行主序可知：连续张量 `[C, T]`（PyTorch）与 ggml 的 `[T, C]` 共享同一内存布局，两侧可将原始字节直接 reshape 对照。

### 2.2 卷积权重展平（量化前提）

ggml 量化张量要求 `ne[0]` 为块大小（32）的整数倍，而 Conv1d/Conv2d 权重的 `ne[0]` 是核宽（3/7/10/11/16…），因此卷积权重无法直接量化；同时量化矩阵乘只允许量化权重位于第一操作数 `src0`，而原生卷积把权重放在第二操作数。解决方案是在导出时把卷积权重展平为二维矩阵：

```
Conv1d [O, I, K]       → (O, I*K)        行序：i 慢、k 快
Conv2d [O, I, KH, KW]  → (O, I*KH*KW)    行序：i 慢、(kh, kw) 快
```

该行序与 im2col 生成的列向量索引顺序一致，展平不需要重排数据。运行时卷积改用 `im2col + 矩阵乘` 实现（见 3.2）。ConvTranspose 权重不展平（原生转置卷积算子只接受 F32/F16 权重）。以元数据布尔键标记该布局，旧布局模型走原生卷积路径。

### 2.3 混合量化规则

| 类别 | 处理 |
|---|---|
| 展平后行宽 % 32 == 0 的大权重（卷积、注意力/FFN 线性） | 按档位量化（q8_0 / q5_1 / q4_1 / q4_0 等） |
| bias、LayerNorm/BatchNorm/GroupNorm 参数、embedding、GRU、`dec.m_source.l_linear`、flow 耦合层、生成器上采样/激励噪声卷积 | 保持 F32 |
| 可量化但行宽不满足或量化失败的张量 | 回退 F16 |

元数据至少包含：网络版本（v1/v2）、采样率、是否 f0、说话人数、`inter_channels/hidden_channels/filter_channels/n_heads/n_layers`、上采样率与核、残差块核与膨胀、`gin_channels`、`spk_embed_dim`、卷积展平标志；RMVPE 另含 `n_fft/hop_length/win_length/n_mels/n_classes/hidden_dim/feature_dim/mel` 边界与 clamp、mel 滤波器组张量。

---

## 三、图执行框架

### 3.1 静态图容器

每个子网络（HuBERT / RMVPE / VITS）以"构建 → 分配 → 执行"三步运行：

```
graph_runner:
    ctx  = init_context()                     # 元数据上下文
    build(ctx)                                # 建图，expand 输出节点
    allocate()                                # 图级内存规划（整体分配输入/中间/输出缓冲）
    set_input(name, host_data) / compute() / get_output(name, host_data)
```

要点：

- 执行采用**单后端直算**（无跨后端调度器），因此所有出现的算子必须被目标后端支持；
- 图按输入形状缓存：同一长度只构建一次，长度变化即重建；
- 常量输入（相对位置嵌入、置换矩阵、kernel 占位张量等）在图分配后上传一次；
- 主机侧输出统一为 F32 行主序数组，便于与任意参考实现对照。

### 3.2 张量布局

| 区域 | 布局 | 原因 |
|---|---|---|
| enc_p / flow（线性、注意力、耦合） | `[C, T]`，C 在 `ne0` | 与 `mul_mat(W, x)` 的自然输入一致；归一化/切片沿 `ne0` |
| dec / 音频前端卷积 | `[T, C]`，C 在 `ne1` | 与 im2col 的自然输入一致，上采样发生在 T 维 |
| 交换点 | enc_p→dec、dec→输出 | 各做一次 `permute + cont` |

---

## 四、算子映射与伪代码

### 4.1 线性层 / 嵌入

```
linear(x, W, b):                      # x: [K, T]，W: [K, N]
    y = mul_mat(W, x)                 # [N, T]（即 Wᵀx）
    if b: y = y + broadcast(b)        # b: [N] → [N, 1]
    return y

get_rows(table, idx):                 # 按整数索引查表（emb_pitch）
    return table[idx]                 # [C, T]
```

### 4.2 Conv1d：展平路径

```
conv1d_flat(x[T, I, B], w[K*I, O], bias[O], stride, pad, dilation):
    kernel = tensor(K, I, 1)                                     # 仅提供形状，im2col 不读其数据
    cols   = im2col(kernel, x, stride, pad, dilation,
                    dst_type = (w.type == F16 ? F16 : F32))      # [K*I, OW, B]
    out    = mul_mat(w, reshape2d(cols, K*I, OW*B))              # [O, OW*B]
    out    = permute_cont(reshape3d(out, O, OW, B), (1,0,2,3))   # [OW, O, B]
    if bias: out += reshape(bias, (1, O, 1))
    return out
```

- `permute` 的参数语义是"源维 i 放到目标轴 aᵢ"；`[O,OW,B]→[OW,O,B]` 为 `(1,0,2,3)`；
- 列向量类型：F16 权重→F16 列；量化或 F32 权重→F32 列（CPU 后端在量化 `src0` 路径要求 `src1` 为 F32）。

### 4.3 Conv2d：展平路径

```
conv2d_flat(x[W, H, IC, N], w[KW*KH*IC, OC], bias, stride, pad):
    kernel = tensor(KW, KH, IC, 1)
    cols   = im2col(kernel, x, stride, pad, 1, 1, is_2d=true, dst_type=…)  # [KW*KH*IC, OW, OH, N]
    out    = mul_mat(w, reshape2d(cols, KW*KH*IC, OW*OH*N))                # [OC, OW*OH*N]
    out    = reshape4d(out, OC, OW, OH, N)
    out    = permute_cont(out, (2,0,1,3))                                  # [OW, OH, OC, N]
    if bias: out += reshape(bias, (1,1,OC,1))                              # 通道维是 ne2
    return out
```

### 4.4 分组卷积（HuBERT 位置卷积，groups=16）

```
grouped_conv(x[C, T], w, g):
    for i in 0..g-1:
        xi = cont(slice_channels(x, i*C/g, (i+1)*C/g))
        wi = cont(slice_channels(w, ...))
        yi = conv2d(xi, wi, …)          # 退化为高度=1 的 2D 卷积
    return concat(yi)
```

### 4.5 转置卷积

```
conv_transpose1d_pytorch(x, w, b, stride, k):
    raw     = conv_transpose_1d(w, x, stride, p0=0)     # 输出两侧各多 pad=(k-stride)/2
    trimmed = view_cont(raw, len = raw.len - 2*pad, offset = pad)
    return trimmed + b

conv_transpose2d_2x(x, w):                              # k=3, s=2, p=1, output_padding=1
    full = conv_transpose_2d_p0(w, x, stride=2)         # 输出 2*in+1
    return cont(view(full, size = 2*in, offset = 1))    # 取 full[1 : 1+2*in]
```

### 4.6 归一化与激活

```
layer_norm(x[C,T]):       # 沿 ne0
    (x - mean_C) / sqrt(var_C + eps) * γ + β

group_norm_hubert(x):     # 逐通道组归一化（groups=C）
    reshape x → [T, 1, C, 1]，对通道维（ne2）执行 group_norm

batch_norm2d_eval(x):     # RMVPE
    (x - running_mean) / sqrt(running_var + eps) * γ + β     # 参数广播到 ne2

gelu(x)      = 0.5x(1+erf(x/√2))        # 精确 erf 版
leaky_relu   = max(x, 0) + slope*min(x, 0)   # 斜率：常规 0.1，生成器末端 0.01
```

### 4.7 多头注意力（逐头小矩阵乘）

```
function mha(x[C, T], Wq, Wk, Wv, Wo, H):
    q = linear(x, Wq); k = linear(x, Wk); v = linear(x, Wv)     # [C, T]
    D = C / H
    for h in 0..H-1:
        qh = cont(view(q, D, T, col_offset = h*D))
        kh = cont(view(k, …)); vh = cont(view(v, …))
        s  = mul_mat(kh, qh) * (1/sqrt(D))                       # [T, T]，softmax 沿 ne0（键维）
        p  = softmax(s)
        oh = mul_mat(transpose(vh), p)                           # [D, T]
    y = linear(concat(oh), Wo)
```

HuBERT 采用 **Post-Norm** Transformer 层（顺序不可交换）：

```
residual = x
x = mha(x) + residual
x = layer_norm1(x)
residual = x
x = fc2(gelu(fc1(x))) + residual
x = layer_norm2(x)
```

enc_p 为**相对位置注意力**（窗口 10）：

```
相对位置嵌入由权重在主机侧展开为两个常量矩阵：rel_k [D, 2T-1] 与 rel_v [2T-1, D]。
s = qᵀk / sqrt(D) + relative_to_absolute( matmul(emb_rel_k, q) )     # [T, T]
p = softmax(s)
ctx = matmul(emb_rel_v, absolute_to_relative(p))                     # [D, T]
```

### 4.8 Flow 逆变换

顺序（逆推理方向）：`Flip, flow6, Flip, flow4, Flip, flow2, Flip, flow0`。每组 flow 由 3 层 WaveNet 组成（核 5），做仿射耦合的逆：

```
x0, x1 = split(x)
h = WN(x0, cond)
x1 = (x1 - h_pos) * exp(-h_log)      # 或加法耦合：x1 = x1 - h
```

通道反转用 **192×192 置换矩阵**与特征做一次矩阵乘实现，替代逐通道切片拼接。

### 4.9 RMVPE（f0）

**mel 前端（主机侧）**：实 FFT → |X|² → mel 滤波器组（由模型携带）→ log → clamp。

**DeepUnet（特征网络）**：5 级编码（每级 4 个 ConvBlockRes + 2×2 平均池化）→ 4 级中间 → 5 级解码（转置卷积上采样 + skip 拼接 + 4 个 ConvBlockRes）→ 输出层 3×3 卷积 → 384 维。ConvBlockRes 为 `conv3x3+BN+ReLU` 两次，残差在通道变化时用 1×1 捷径。

**整段与窗口**：整段推理时图按 padded 帧数构建；超长时退回窗口推理——窗口有效帧 4096、上下文各 2048（图固定 8192 帧），拼接只取有效区。Vulkan 下整段上限为 16352 帧（推导见 §6.2）。

**双向 GRU（512 步静态展开 + 状态接力）**：

```
function gru_chunk(x[384, 512], keep[512], h0[256]):
    xp = linear(x, W_ih) + b_ih                  # [768, 512]
    h = h0
    for t in 0..511:
        g = xp[:, t] + linear(h, W_hh) + b_hh    # [768]
        r, z, n = split(g, [256, 256, 256])      # 门序 r/z/n 与 PyTorch 一致
        n = tanh(n)
        h = (1 - z) * n + z * h
        seq[t] = h
    return seq, h
```

- 前向按 512 帧分块，块尾隐状态传给下一块；反向从末尾向前分块（块内按逆序展开），同样接力；
- `keep` 标记有效帧（补齐帧不参与状态更新）。

**显著性头**：`[fwd; rev]`（512 维）→ Linear(360) → Sigmoid。

**解码**：

```
idx(t) = argmax_c salience[c, t]
cents(t) = Σ_{k=-4..4} w_k * (20*(idx(t)+k) + 1997.3794084376191)   # 局部平均
f0(t) = 10 * 2^(cents(t)/1200)
if salience[idx(t), t] < 0.03: f0(t) = 0                            # 清音
```

### 4.10 检索（FAISS IVF 等价语义）

```
function ivf_blend(feat, index, index_rate, k=8):
    lists = 按 L2² 质心距离取最近的 nprobe 个倒排列表        # nprobe 随索引存储
    cands = 拼接各列表的向量
    topk  = L2² 距离最小的 k 个候选
    w     = (1 / score)²，并归一化
    mix   = Σ w_i * vec_i
    return index_rate * mix + (1 - index_rate) * feat
```

索引导出为四个张量：质心、全部向量、各列表偏移、各列表长度；混合发生在 ×2 最近邻上采样之前。

### 4.11 生成器（HiFiGAN NSF）与 SineGen

```
dec(z[C, T], g, source):
    x = permute_cont(z) → [T, C]
    x = conv1d(x, dec.conv_pre) + broadcast(dec.cond(g))     # 条件项
    for i in 0..n_ups-1:
        x = leaky_relu(x)
        x = conv_transpose1d_pytorch(x, dec.ups[i])
        if f0: x += noise_conv_i(source)                     # 见下
        x = (Σ_j resblock_j(x)) / n_resblock                 # 每级 3 个残差块，核/膨胀 {k,1},{k,3},{k,5}
    x = leaky_relu(x, 0.01)
    x = conv1d(x, dec.conv_post)                             # 1 通道
    return tanh(x)
```

- 每级上采样后的激励噪声卷积步长 = 其余上采样率之积（1×1 特例：核 1、步长 1），核长为 `2*stride`（末级 1）；
- 残差块内：`convs1` 带膨胀（pad=(k*d-d)/2），`convs2` 核 3 无膨胀。

SineGen（主机侧逐样本）：

```
rad(t)   = (f0(t)/sr) mod 1
cum(t)   = cumsum(rad)
phase(i) = cumsum_shift(线性插值(cum*upp, align_corners=True))      # 相位回绕补偿
voiced   = f0 > 0
noise_amp = voiced ? 0.003 : 0.1/3
out(i)   = voiced ? [sin(2π*phase(i)) * 0.1 + noise_amp*noise(i)]
                  : [noise_amp * noise(i)]
```

其中 `upp = sr / 100`，`noise` 可由固定种子生成或外部注入；输出再经 `tanh(l_linear(·)+b)` 作为激励源。

---

## 五、全链路流水线

### 5.1 预处理

1. **下混 + 重采样**：任意输入 → 单声道 16 kHz，使用多相 FIR（有理比重采样）。
2. **48 Hz 零相位高通**：二阶 IIR 前向/反向各滤一次（filtfilt），边界采用奇扩展，`padlen = 3*max(len(a),len(b))`。
3. **整段 f0**：对反射补齐后的音频跑 RMVPE；f0 乘变调系数 `2^(pitch/12)`，并量化为 1..255 的
   `pitch_bin = 1 + round(12*log2(f0/50))`（清音为 0）。

### 5.2 切块与拼接（与 Python 参考一致）

```
function split_points(audio, x_query, x_center, x_max):      # 单位：秒
    if len(audio)/16000 <= x_max: return []
    curve = Σ_{i=0..159} |audio_pad[t+i]|                    # 滑动 160 点绝对和，反射 pad 80
    for t = x_center*16000 step x_center*16000:
        lo = max(0, t - x_query*16000); hi = min(len, t + x_query*16000)
        emit(t - x_query*16000 + argmin(curve[lo:hi]))
```

拼接规则：

```
pad = x_pad*16000
for t in split_points:
    t = align_to_10ms(t)
    chunk = padded[s : min(t + 2*pad + 160, len)]
    f0_slice = f0[s/160 : (t + 2*pad)/160]
    emit(vc(chunk, f0_slice))
    s = t
emit(vc(padded[s:], f0[s/160:]))
```

### 5.3 单块转换（vc）

```
T_tok, C = hubert(audio_chunk)                # 50 fps
feats0   = feats                              # protect < 0.5 时保留副本
if index: feats = ivf_blend(feats, index_rate)
feats = upsample2_nearest(feats)              # 100 fps（×2，最近邻）
p_len = min(len(audio_chunk)/160, feats.frames, len(f0_slice))

for t in 0..p_len-1:                          # protect（清音保护）
    pf = (f0[t] >= 1 Hz) ? 1.0 : protect
    feats[t] = pf * feats[t] + (1-pf) * feats0[t]

source = tanh(l_linear * sinegen(f0_slice) + b)
audio  = vits(feats, pitch_bin_slice, sid, source, noise)
trim   = x_pad * target_sample_rate
return audio[trim : len-trim]
```

### 5.4 后处理

```
# RMS 响度混合
rms_src = 160 点帧 RMS(原 16k)，rms_conv = 160 点帧 RMS(转换结果，目标采样率)
factor(i) = rms_src(i)^(1-r) * rms_conv(i)^(r-1)
out *= factor（帧间线性插值到逐样本）

# 峰值归一化与 int16
peak  = max|out|
scale = 32768; if peak/0.99 > 1: scale /= peak/0.99
pcm   = int16(out * scale)
```

---

## 六、后端约束与适配

### 6.1 ggml 量化的硬约束

1. 量化张量 `ne[0]` 必须是 32 的整数倍；
2. `mul_mat(A, B)` 中量化只允许出现在 `A`（`src0`）；
3. CPU 后端在 `src0` 量化时要求 `src1` 为 F32；
4. Vulkan 后端不支持"量化 `B`"（`src0` 非 F16/F32 且 `src1` 为量化类型时断言），故量化权重必须在 `src0`。

### 6.2 Vulkan 网格上限与 matmul 的 N 维切分

Vulkan 的 compute dispatch 每个维度受 `maxComputeWorkGroupCount` 限制（各维最低保证 65535）。矩阵乘按
`ceil(N / wg_denoms[1])` 生成 Y 维 workgroup，本项目相关路径 `wg_denoms[1] = 32`，因此 `N > 65535×32`
时会触发后端断言。原版 ggml 只对超大 **M**（受 `maxStorageBufferRange` 限制）切分，N 维不切分；实现中对其补充 N 切分：

```
function mul_mat(dst, a, b):
    max_n = device.maxComputeWorkGroupCount[1] * 32        # 32 = 所有 matmul pipeline 中最小的 wg_denoms[1]
    if (dst.ne2 == 1 and dst.ne3 == 1 and dst.N > max_n):
        for n0 in 0 .. dst.N step max_n:
            cn   = min(max_n, dst.N - n0)
            dst' = view(dst, N=cn, offset += n0 * nb1)
            b'   = view(b,   N=cn, offset += n0 * nb1)
            matmul_shader(a, b', dst', disable_split_k = true)
        return
    ……原路径（含超大 M 的按行切分）
```

要点：分块步长取最小 `wg_denoms[1]` 以保证任意 pipeline（s/m/l、F16/F32/量化）都不超限；分块内禁用 split-k，避免其归约 dispatch 的元素数 `m×n×batch` 再次超限；仅 batch=1 生效。

由该约束反推两个实现阈值：

| 阶段 | N 的构成 | Vulkan 安全上限 |
|---|---|---|
| VITS 生成器末级卷积 | `VITS 帧数 × hop`（hop=400/480） | hop=480 时单块 ≤ 4369 帧；hop=400 时 ≤ 5242 帧 |
| RMVPE 整段卷积 | `128 × 帧数` | 帧数 ≤ 16352（约 163.5 s），超限改走窗口路径 |

### 6.3 Vulkan 算子支持要点

| 算子 | 约束 |
|---|---|
| `IM2COL` | 支持 F32/F16 输出；配合 `mul_mat` 实现展平卷积 |
| `CONV_TRANSPOSE_1D` | 仅 F32 权重与输出（生成器上采样强制 F32） |
| `CONV_2D / CONV_TRANSPOSE_2D` | 权重/输入/输出需连续，通道数须匹配；仅旧布局路径使用 |
| `CONCAT` | 两侧同类型、连续（切片后先 `cont`） |
| `MUL_MAT` | 量化 `src0` 支持 q4_0/q4_1/q5_1/q8_0；N 超大时走 §6.2 切分 |

默认关闭两项加速路径以规避驱动问题：整数点积（Intel 驱动在量化多次图执行后可能输出 NaN/零）与 fp16 计算（误差放大，音频 SNR 由约 60 dB 降至约 33 dB）。均可通过环境变量恢复。

### 6.4 其他适配点

- `concat/repeat` 等对连续性敏感：任何切片/视图参与前先做连续化；
- 着色器首次运行需编译 SPIR-V（数秒），驱动缓存后启动加快；
- 权重常驻显存，图缓冲按图规模分配；块越大显存占用越高。

---

## 七、数值与确定性

| 项 | 说明 |
|---|---|
| 精度 | 全流程 float32；部分原生卷积内部 im2col 为 fp16，误差约 1e-3 量级 |
| f0 敏感性 | 显著性 top1/top2 接近时 argmax 可能翻转（<0.1% 帧）；单帧错误会造成 SineGen 相位偏移，需按帧级纠错/中值滤波抑制 |
| 随机噪声 | 默认固定种子（块序号参与派生），保证可复现；可注入外部噪声实现逐样本对照 |
| 重采样 | 多相 FIR 与 soxr 为质量级一致，非逐位一致；默认不做输出重采样时不受影响 |
| 检索距离 | 与 FAISS SIMD 实现可能存在末位差异，混合权重一致 |
| 对齐基准 | `mul_mat(A,B) = AᵀB`、`softmax/norm` 沿 `ne0`、`group_norm` 通道维为 `ne2`、GELU 用 erf 精确版——这些语义决定移植时的轴选择与顺序 |
