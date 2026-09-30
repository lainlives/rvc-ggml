# RVC-GGML

纯 C++ / [ggml](https://github.com/ggml-org/ggml) 实现的 [RVC（Retrieval-based-Voice-Conversion）](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)**训练 + 推理**，无需 CUDA——**AMD / Intel / NVIDIA 显卡通过 Vulkan 均可加速**，无独显也可 CPU 运行。

- **rvc-ggml**：推理。把 Python/PyTorch 的 RVC 推理链路（HuBERT 特征 → RMVPE 变调 → VITS 合成 → FAISS 检索）完整移植为 C++ + ggml，模型以 GGUF 格式分发。
- **rvc-train-ggml**：训练。基于 [train.cpp](https://github.com/Chui-Las/train.cpp)（C++ 深度学习训练库，autograd + Vulkan 后端）实现 RVC 的预处理、GAN 训练循环与推理模型导出，与 rvc-ggml 形成"训练 → 导出 GGUF → 推理"的闭环。

> 设计目标：输出与 Python 原版数值/听感一致（导出模型严格自检 corr=1.0000，端到端 corr ≥ 0.999）。

## 仓库结构

```
├── rvc-ggml/          推理（RVC-GGML.exe + Python 模型转换脚本）
├── rvc-train-ggml/    训练（rvc-prep.exe 预处理 + rvc-train.exe 训练/导出）
└── third_party/
    └── train.cpp/     训练库（git submodule）
```

## 系统要求

- Windows 10/11 x64（当前验证的主平台）
- CMake ≥ 3.20、MSVC（Visual Studio C++ 工具集）、Ninja
- [Vulkan SDK](https://vulkan.lunarg.com/)（含 glslc；不装则自动回退纯 CPU）
- 支持 Vulkan 1.2 的 GPU（显存/共享内存 ≥ 4GB 起步）；无 GPU 可用 CPU 后端

## 构建

```powershell
git clone --recursive https://github.com/Chui-Las/RVC-GGML.git RVC-GGML
cd RVC-GGML

# 训练侧（会级联构建 rvc-prep / rvc-train / RVC-GGML）
powershell -ExecutionPolicy Bypass -File rvc-train-ggml\scripts\dev_build.ps1
# 产物在 rvc-train-ggml\build\bin\
```

或手动 CMake：

```powershell
cmake -S rvc-train-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

ggml 依赖自动解析：FetchContent 拉取官方 ggml v0.23.0 后自动应用本项目补丁
（见 `rvc-ggml/patches/`）；也可用 `-DRVC_GGML_DIR=<本地 ggml 源码>` 指定本地副本。

## 快速上手

### 1. 模型转换（.pth → GGUF）

需要原版 RVC 生态的模型文件（hubert_base.pt、rmvpe.pt、角色 .pth/.index）：

```powershell
cd rvc-ggml
python scripts\export_gguf.py hubert --input <RVC目录>\assets\hubert\hubert_base.pt --out models\hubert.gguf --dtype fp16
python scripts\export_gguf.py rmvpe  --input <RVC目录>\assets\rmvpe\rmvpe.pt       --out models\rmvpe.gguf  --dtype q8_0
python scripts\export_gguf.py model  --input <角色.pth>                             --out models\<角色>.gguf --dtype q4_0
python scripts\export_gguf.py index  --input <角色.index>                           --out models\<角色>.index.gguf
# 或从原版 RVC 根目录一键批量：
python scripts\export_gguf.py all --rvc-root <RVC目录> --out models --dtype fp16
```

### 2. 推理

```powershell
RVC-GGML.exe --input input.wav --model models\<角色>.gguf \
             --index models\<角色>.index.gguf --index-rate 0.75 \
             --model-dir models --device 0 --output output.wav
# --device 0 = Vulkan（默认列出后选择），99 = CPU
RVC-GGML.exe --list-devices
# 不带参数启动进入交互式向导；模型训练参数（变调/检索率等）内嵌于角色 GGUF
```

### 3. 训练

微调官方预训练模型（最小闭环）：

```powershell
cd ..\rvc-train-ggml
# 0) 官方预训练 G/D .pth → 训练态 GGUF（一次性，见 tools\README.md）
python tools\pth_to_train_gguf.py --pth <RVC目录>\assets\pretrained\f0G40k.pth --config configs\v1-40k-f0.json --out work\f0G40k.train.gguf
python tools\pth_to_train_gguf.py --pth <RVC目录>\assets\pretrained\f0D40k.pth --config configs\v1-40k-f0.json --out work\f0D40k.train.gguf

# 1) 预处理 + 训练 + 导出 + 闭环推理，一条命令：
powershell -ExecutionPolicy Bypass -File scripts\train_pipeline.ps1 `
    -Audio "D:\myvoice" -Mode finetune `
    -G work\f0G40k.train.gguf -D work\f0D40k.train.gguf `
    -Epochs 50 -ExportEvery 5 -Export -Infer -InferInput test.wav

# 或交互式向导（双击 exe / 不带参数运行）：
rvc-train.exe --interactive
# 从零训练：向导中选择"从零"，或 rvc-train --scratch --train configs\v1-40k-f0.json
```

> 注意：预处理（rvc-prep，ggml 后端）与训练（rvc-train，traincpp 后端）是两个独立进程，
> 由 train_pipeline.ps1 / 向导自动串联，不能在一个 exe 内完成全部步骤。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`rvc-ggml/docs/使用文档_发布版使用与原版对比.md`](rvc-ggml/docs/使用文档_发布版使用与原版对比.md) | 推理使用说明、参数对照、常见报错 |
| [`rvc-ggml/docs/技术文档四_源码结构与模块手册.md`](rvc-ggml/docs/技术文档四_源码结构与模块手册.md) | 推理侧源码结构与接口手册 |
| [`rvc-ggml/docs/技术文档三_统一实现细节_Python转ggml.md`](rvc-ggml/docs/技术文档三_统一实现细节_Python转ggml.md) | 算子映射与流水线数学定义（自包含） |
| [`rvc-ggml/docs/技术文档二_ggml_Vulkan实现细节.md`](rvc-ggml/docs/技术文档二_ggml_Vulkan实现细节.md) | GGUF 布局、量化、Vulkan 适配与坑 |
| [`rvc-ggml/docs/技术文档一_项目移植与实现概览.md`](rvc-ggml/docs/技术文档一_项目移植与实现概览.md) | 推理移植概览与功能对照 |
| [`rvc-train-ggml/docs/使用指南.md`](rvc-train-ggml/docs/使用指南.md) | 训练全流程使用说明 |
| [`rvc-train-ggml/docs/数字手册.md`](rvc-train-ggml/docs/数字手册.md) | 训练参数/日志数字判读 |
| [`rvc-train-ggml/docs/架构设计.md`](rvc-train-ggml/docs/架构设计.md) | 训练侧架构、数据流、GGUF schema、缓存格式 |

## 兼容性说明

- 部分旧 Intel 驱动在量化模型多次图执行时整数点积路径有缺陷（NaN/全零），程序默认规避，
  可用 `RVC_VK_ENABLE_INTDOT=1` 恢复；fp16 计算路径默认同样禁用（`RVC_VK_ENABLE_F16=1` 恢复）。
- 官方 ggml 的 Vulkan 矩阵乘在 N 维超限时（长音频 / 48kHz 模型）会触发
  `maxComputeWorkGroupCount` 断言——本项目补丁已修复，FetchContent 路径自动应用。

## License

本项目以 [MIT](LICENSE) 协议开源。

本项目依赖/参考了以下优秀的开源项目，各自许可证见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)：

- [ggml](https://github.com/ggml-org/ggml)（MIT）—— 推理后端
- [train.cpp](https://github.com/Chui-Las/train.cpp)（MIT）—— C++ 深度学习训练库（submodule）
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)（MIT）—— 语义基准与模型生态
- [dr_wav](https://github.com/mackron/dr_libs)（MIT-0 / Unlicense）与
  [pocketfft](https://github.com/mreineck/pocketfft)（BSD-3）—— vendored 单头库

