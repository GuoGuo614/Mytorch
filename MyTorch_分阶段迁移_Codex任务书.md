# MyTorch 分阶段迁移：Codex 任务书

目标仓库：[GuoGuo614/Mytorch](https://github.com/GuoGuo614/Mytorch)
接口参考：[GuoGuo614/MyTorch-1](https://github.com/GuoGuo614/MyTorch-1)

## 使用方式

1. 严格按 `V0 → V12` 执行；每一版验收通过后再开始下一版。
2. 第一次先把“总控指令”发给 Codex，再发送当前版本任务。如果换了 Codex 会话，应把总控指令重新发一次。
3. 每版完成后先看 `git diff`、测试结果和未验证项，再由你决定是否提交。任务书默认要求 Codex **不自动 push**。
4. M1 Mac 只负责 CPU 测试；CuPy、Triton、NCCL 必须在 NVIDIA CUDA 机器上验收。没有对应硬件时，Codex 必须明确写“未做真实硬件验证”，不能用 mock 冒充通过。

## 总控指令（首次发送）

```text
你正在维护当前仓库 GuoGuo614/Mytorch。目标是保留这个项目简单、适合教学的 TensorOp/Op.compute/Op.gradient 自动微分架构，分阶段吸收 GuoGuo614/MyTorch-1 中少量有价值的接口与实现思路，而不是把复杂仓库整体复制过来。

固定约束：
1. 当前仓库是唯一修改目标；MyTorch-1 只读参考。修改前先检查 git status，不覆盖或回滚用户已有改动。
2. canonical framework 是原 my_conv_proj/mytorch；hw4/python/needle 仅作为 legacy/reference，不得把两个框架混合导入，也不得整体复制 hw4 的 NDArray/autograd。
3. 保留 TensorOp.compute/gradient 的自动微分模型。可以参考 MyTorch-1 的 public API、设备抽象、kernel dispatch 和 apps 组织方式，但不要替换成它的 grad_fn 闭包体系。
4. NumPy CPU 是无条件可用的基础后端；CuPy、Triton、NCCL 都必须是可选依赖并懒加载。CPU-only 环境不得因 import cupy/triton 失败。
5. 不引入 PyTorch 作为运行时、张量分配器、自动微分器或测试 oracle 的硬依赖；数值基准优先使用 NumPy/CuPy。若测试临时使用 torch，只能放在可选测试依赖中并说明原因。
6. CPU/GPU 张量不得静默混算；device、dtype、shape 和 contiguous 限制必须明确检查。不能在热路径中偷偷把 GPU 数组搬回 CPU。
7. 每个新增优化都必须有正确性回退路径、单元测试和 benchmark；不要硬编码 PPT 中的性能数字。
8. 不提交数据集、checkpoint、构建产物、__pycache__、视频或大体积 benchmark 日志。
9. 对从 MyTorch-1 或其上游改写的代码保留 MIT 许可和来源说明；不要把“参考改写”描述成完全原创。
10. 一次只完成我指定的版本。完成后停止，不要自行开始下一版，不要 push。

每一版结束时输出：
- 改动文件清单与关键设计决定；
- 实际执行的测试/benchmark 命令及结果；
- 在当前机器上未验证的项目；
- git diff --stat；
- 下一版开始前需要我决定的问题。
```

## 版本路线图

| 版本 | 交付目标 | 覆盖的答辩内容 |
|---|---|---|
| V0 | 整理包结构并冻结 CPU 基线 | 框架可安装、可复现 |
| V1 | Device 与 NumPy/CuPy 数组后端 | CuPy GPU 后端 |
| V2 | 全框架设备化 | CPU/GPU 同一套接口 |
| V3 | Conv2d naive 与 im2col+GEMM | 卷积优化 |
| V4 | Triton 融合 Linear/Softmax/Norm | 重要算子融合 |
| V5 | Triton Conv 路径 | Triton 卷积 |
| V6 | 异步 DataLoader | 生产者/消费者预取 |
| V7 | ResNet 双头 AutoDrive 训练 | ResNet、转向与油门 |
| V8 | 数据采集、推理与 Grad-CAM | 多地图、动态油门、可解释性 |
| V9 | 最小 Transformer Encoder | Transformer 扩展 |
| V10 | 单机多卡数据并行 | 并行训练、NCCL |
| V11 | 训练监控面板 | 可视化前后端 |
| V12 | Docker、基准与最终文档 | GPU 部署、答辩证据闭环 |

---

## V0：整理工程并冻结 CPU 基线

```text
执行迁移 V0：整理工程并冻结 CPU 基线。只完成本版。

目标：让仓库从根目录可安装、可测试，同时保留现有 CPU 行为，给后续 GPU 改造建立可信基线。

任务：
2. 使用 git mv 将 my_conv_proj/mytorch 移到根目录 mytorch/，更新现有 app 与测试导入；不要通过 sys.path hack 解决导入。
3. 新增根目录 pyproject.toml。基础依赖只含 NumPy 等 CPU 必需项；预留 dev、cuda、triton、dashboard 等 optional extras，但本版不实现 GPU。
4. 新增 README：架构概览、CPU 安装、最小 Tensor/autograd 示例、MNIST 示例入口、仓库目录说明。
5. 新增或完善 .gitignore，忽略 __pycache__、构建产物、数据集、checkpoint、日志。已跟踪的大数据文件先列出，不要擅自删除。
6. 新增 AGENTS.md，写入总控指令中的架构边界和逐版验收规则，便于后续 Codex 会话遵守。
7. 建立 tests/：至少覆盖 Tensor 基本运算、拓扑反传、Linear 前后向、Conv2d 小尺寸前后向，以及一个不下载数据的 tiny training smoke test。
8. 修复因目录整理产生的问题，但不要重写 autograd 或优化算子。

验收：
- 在干净 CPU Python 环境执行 pip install -e '.[dev]' 成功；
- python -c 'import mytorch' 成功；
- pytest -q 全部通过；
- 原有 LeNet/MNIST 脚本至少能通过 --help 或 tiny synthetic smoke test；
- git diff 中不包含数据集、pyc、checkpoint；
- 不自动 commit，不 push，完成后停止。
```

## V1：实现 Device 与 NumPy/CuPy 数组后端

```text
执行迁移 V1：实现最小 Device 与 NumPy/CuPy 数组后端。只完成本版。

先阅读当前 mytorch 的 Tensor/autograd/backend，再只读参考 MyTorch-1 的数组后端与 Tensor 设备接口。保留当前 TensorOp.compute/gradient 架构，不复制复杂版 Tensor/autograd。

任务：
1. 新增清晰、很小的数组后端模块，提供 cpu()、cuda(index=0)、is_cuda_available()、get_array_module(x)、asarray、asnumpy、to_device，以及 zeros/ones/empty/randn 等必要工厂。
2. CuPy 必须懒加载；未安装 CuPy 时 import mytorch 和全部 CPU 测试仍能通过。cuda() 在不可用时给出明确异常与安装提示。
3. 为 Tensor 增加 device、dtype、to(device)、cpu()、cuda(index)、numpy()；创建 Tensor 时保持输入 dtype，不做无说明的 float64/float32 转换。
4. 禁止 CPU 与 GPU Tensor 静默参与同一运算；错误信息要包含两个 device。
5. 设计 public API 时尽量与 MyTorch-1 同名同语义，但内部仍由当前 Op.compute/gradient 驱动。
6. pyproject 的 cuda extra 使用适配 CUDA 12 的 CuPy 包；README 分开写 CPU、NVIDIA CUDA 安装方式，并说明 Apple Silicon 不能运行 CUDA/CuPy 路径。

测试：
- CPU：构造、to/cpu/numpy、dtype/device、不可用 CUDA 报错；
- NVIDIA：CuPy 数组创建、CPU↔GPU 显式传输、同设备简单运算；
- 验证 import mytorch 不会主动 import cupy；
- 本版只让最小算术链路设备可感知，不要顺手重写所有算子，那是 V2。

完成后报告真实执行结果；若没有 NVIDIA 环境，明确列出 GPU 测试为未验证，不得伪造。不要 commit/push，停止。
```

## V2：将算子、nn、初始化器和优化器全面设备化

```text
执行迁移 V2：让主框架在同一套 API 下完整支持 NumPy CPU 和 CuPy GPU。只完成本版。

任务：
1. 审计 mytorch 下所有 ops、nn、init、optim、data 和 app，消除 compute/gradient 热路径中硬编码的 numpy；根据输入数组选择 xp，并确保新建数组位于正确 device。
2. 所有 Parameter、Module.to(device)、state_dict/load_state_dict、初始化器和 Optimizer 更新必须保持 device 与 dtype。
3. 为 Module 实现递归 to(device)；Sequential、Residual、Linear、归一化、激活、reshape/transpose/broadcast/reduction、loss、optimizer 至少完整可用。
4. 不允许用 asnumpy/.get() 规避 GPU 算子实现；仅日志、序列化或显式 Tensor.numpy() 可以传回 CPU。
5. 对现有不支持 CuPy 的算子，要么正确实现，要么抛出带算子名的 NotImplementedError；不能静默 fallback 到 CPU。
6. 增加 CPU/GPU 参数化测试：同一随机输入比较 forward、backward 和一次 optimizer step。容差按 dtype 合理设置。
7. 新增 examples/device_smoke.py 或等价脚本，在 CPU 与 cuda:0 上训练同一个小 MLP 若干步并比较 loss 下降。

验收：
- CPU 全套测试继续通过；
- NVIDIA 上核心 MLP 和小 Conv 网络 forward/backward/step 成功；
- 没有隐式 host-device copy；
- 不引入 PyTorch 运行时；
- 不开始卷积优化或 Triton 融合。不要 commit/push，停止。
```

## V3：Conv2d 的 naive 与 im2col+GEMM 双路径

```text
执行迁移 V3：为 Conv2d 增加可验证的 naive 与 im2col+GEMM 实现。只完成本版。

任务：
1. 保留当前 Conv2d public API、输入布局和权重布局；若布局不清晰，先补文档和断言，不要静默改格式。
2. 将现有 Python 循环卷积保留为 implementation='naive' 的正确性基线。
3. 实现 implementation='im2col'：forward 使用向量化 im2col + 后端 matmul/GEMM；backward 正确实现 dX、dW、db，并在 NumPy/CuPy 后端均可运行。
4. 增加 implementation='auto'，当前阶段在支持的形状上选 im2col，否则回退 naive；选择逻辑必须可测试、可记录。
5. 支持当前框架已经承诺的 stride、padding、bias 与非方形输入；暂不扩展 dilation/groups，除非原 API 已经支持。
6. 控制临时内存，至少对过大的 im2col 矩阵给出说明或分块策略，不能无界申请显存。
7. 新增 tests/test_conv_impls.py：多种 batch/channel/kernel/stride/padding/非方形组合比较 naive 与 im2col 的 forward 和梯度。
8. 新增 benchmarks/bench_conv.py，输出形状、实现、设备、预热次数、中位数/P95 和峰值内存；不写死结果。

验收：
- CPU 数值测试全过；
- CUDA 环境下 CuPy im2col 路径全过；
- benchmark 能显示 naive 与 im2col 的真实数据；
- 本版不写 Triton Conv。不要 commit/push，停止。
```

## V4：迁移四个关键 Triton 融合算子

```text
执行迁移 V4：增加少量、重要且可回退的 Triton 融合算子。只完成 Linear、Softmax、LayerNorm、RMSNorm，不做其他模型或 FlashAttention。

先在 MyTorch-1 中定位相关 kernel、dispatch 和 public API，理解后适配当前 TensorOp.compute/gradient 架构。保留许可证与来源说明，不整体复制复杂框架。

任务：
1. 新建 mytorch/kernels/，分别组织 linear、softmax、layernorm、rmsnorm 的 Triton kernel 和 dispatch。
2. Triton 与 CuPy 均为 optional extra 并懒加载；CPU、无 Triton、形状不支持、dtype 不支持时走已有正确实现。
3. public API 尽量与 MyTorch-1 一致：用户仍调用普通 nn.Linear、softmax、nn.LayerNorm、nn.RMSNorm，通过 implementation='auto'|'eager'|'triton' 或等价的清晰开关选择。
4. CuPy Tensor 直接作为 CUDA 数据来源；不要为调用 Triton 引入 torch/DLPack 中转依赖。
5. forward 和 backward 都必须数值正确。若某个 backward 暂时使用 eager 公式，必须在代码、文档、benchmark 中明确标注，不能宣称整套训练均融合。
6. 对 contiguous、shape、axis、block size、dtype、CUDA capability 做显式检查；强制 triton 时不支持应清晰报错，auto 时才允许回退。
7. 测试覆盖不规则尺寸、小尺寸、较大尺寸、float32，并在硬件支持时覆盖 float16；比较 eager 与 triton 的 forward/gradient。
8. 新增 benchmarks/bench_fused_ops.py，含预热与 GPU synchronize，报告延迟和峰值显存。

验收：CPU-only import/test 不受影响；RTX 4060 上四个算子均有至少一个真实 forward/backward 用例；没有 PyTorch 运行时依赖。不要 commit/push，停止。
```

## V5：Triton Conv 路径

```text
执行迁移 V5：在 V3 Conv 双路径上增加 Triton Conv。只完成本版。

任务：
1. 只读参考 MyTorch-1 的 Conv kernel/dispatch，适配当前 TensorOp 架构和 V3 的 Conv2d API。
2. 增加 implementation='triton'，并让 'auto' 只在 CUDA、dtype、布局和形状满足条件时选择 Triton；其他情况回退 im2col。
3. 先保证 forward 正确；backward 若使用 im2col/eager 回退必须明确标注。只有实际实现并测试了 Triton backward，才可称“卷积训练前后向均 Triton 加速”。
4. 不支持的 stride/padding/kernel/dtype 要有明确 dispatch 规则；不能得到错误结果后才回退。
5. 测试比较 naive、im2col、triton 的 forward，并比较完整 Conv2d autograd 梯度；覆盖非整块尺寸和边界 padding。
6. benchmark 同时报告三条路径，包含预热、同步、中位数/P95 和峰值显存；至少选择小、中、大三类形状。
7. 更新 README 的支持矩阵，准确区分 Triton forward 与 backward 的覆盖范围。

没有 NVIDIA/Triton 环境时可以完成静态实现和 CPU 回退测试，但必须把真实 GPU 验收列为 blocker，不得用跳过测试声称完成。不要 commit/push，停止。
```

## V6：异步 DataLoader 与 CIFAR-10 基准

```text
执行迁移 V6：实现可关闭、可复现、能正确传播异常的异步 DataLoader。只完成本版。

任务：
1. 保留同步 DataLoader 为 num_workers=0 基线；新增 producer/consumer + 有界 Queue 的预取模式，提供 num_workers、prefetch_factor、shuffle、drop_last、seed。
2. worker 中的异常必须传回主线程/进程并清理所有 worker；迭代提前退出、KeyboardInterrupt 和 epoch 结束都不能泄漏后台线程或死锁。
3. 保证固定 seed 下同步与异步的样本集合/顺序可复现；不要让预取改变 epoch 边界。
4. 可选支持 pinned host memory 和批量传入 GPU，但不能让 CPU-only 安装依赖 CuPy。明确 host→device copy 发生位置。
5. 增加轻量 CIFAR-10 dataset/transform 接口和下载说明；测试使用 synthetic dataset，不在仓库提交数据。
6. tests 覆盖顺序、shuffle seed、drop_last、多 epoch、worker 异常、提前 break、队列容量和资源回收。
7. benchmarks/bench_dataloader.py 对比同步/异步吞吐和一个小 CNN epoch 时间，输出环境与配置；不得复制 PPT 的 735s/480s 数字，只有真实复现后才记录新结果。

完成后分别报告 CPU 与 CUDA 端实际测试；不要顺手实现 dashboard。不要 commit/push，停止。
```

## V7：ResNet 双输出 AutoDrive 训练链路

```text
执行迁移 V7：建立 apps/autodrive 的 ResNet 多任务训练链路。只完成 dataset、model、train、evaluate 和 checkpoint，不做模拟器控制或 Grad-CAM。

先只读参考 MyTorch-1 的 apps/auto-drive 目录和接口风格，但不要照搬其中固定油门、单输出五层 CNN 的限制。

任务：
1. 补齐主框架构建小型 ResNet 所需的最少 nn 组件，例如 Conv-BN-ReLU residual block、BatchNorm2d、global/adaptive average pooling；不要迁移庞大 model zoo。
2. 在 apps/autodrive/model.py 实现轻量 ResNet backbone + 两个 head：steering 与 throttle。steering 输出范围清晰（如 tanh），throttle 输出范围可配置并有物理上下界。
3. 统一数据 manifest，至少包含 image_path、steering、throttle、map_name、split/run_id；禁止继续只从文件名解析 steering。
4. Dataset 支持图像预处理和只对训练集生效的数据增强；按 run/map 切分 train/val，避免相邻帧泄漏。
5. 训练 loss 为 steer_loss + lambda_throttle * throttle_loss；日志至少记录总 loss、两个子 loss、学习率、epoch 时间和验证指标。
6. checkpoint 保存 model、optimizer、epoch、配置与归一化统计；提供明确 resume 行为。格式优先 npz/JSON 或可选 safetensors，不依赖 PyTorch。
7. 提供 synthetic tiny dataset smoke test，验证 forward shape、两个 head 的梯度、一步训练、保存/恢复与 loss 下降。

验收：CPU smoke test 必过；CUDA 上至少完成一次小批次 forward/backward/optimizer step。不要宣称已达到 PPT 的圈速或完成率，那需要 V8 后的真实实验。不要 commit/push，停止。
```

## V8：自动驾驶采集、闭环推理与 Grad-CAM

```text
执行迁移 V8：补齐 AutoDrive 的数据采集、闭环推理和 Grad-CAM。只完成本版。

任务：
1. 统一 apps/autodrive/config 与命令行入口，明确 simulator adapter；把模拟器依赖隔离，保证无模拟器时模块仍可 import 和测试。
2. 数据采集器同时记录 steering 与 throttle：A/D 控制转向，W 提升油门，松开后缓慢衰减到 base throttle；所有参数可配置并写入 metadata。
3. 支持 generated-track、mountain-track、warren-track、warehouse 等 map_name；manifest 记录 map、run_id、时间戳、图像路径、steering、throttle。不要只把标签编码进文件名。
4. 提供数据校验/汇总脚本：坏图、缺失标签、重复帧、每地图样本数、steering/throttle 分布、8:2 group split。
5. 闭环推理同时使用模型预测 steering 和 throttle；训练数据没有 throttle 时，固定为 0.2
6. 实现针对 steering head 的 Grad-CAM：使用最后一个卷积特征和框架自身 autograd，输出叠加图；可选支持 throttle head。不要为了 Grad-CAM 引入 PyTorch hooks。
7. 加 mock simulator 单元测试，验证动态油门状态机、manifest、控制裁剪、异常降级和 Grad-CAM 输出尺寸；真实模拟器实验单独记录。
8. 更新 apps/autodrive/README：采集→校验→训练→评估→驾驶→Grad-CAM 的完整命令。

完成后把“代码测试”和“真实赛道验证”分开报告；没有模拟器实测时不得复用 PPT 的 70%/100% 或圈速数字。不要 commit/push，停止。
```

## V9：最小 Transformer Encoder

```text
执行迁移 V9：为答辩中的 Transformer 扩展实现一个最小、教学友好的 Encoder，不迁移 GPT/nanochat/FlashAttention/KV cache/MoE。只完成本版。

任务：
1. 在主框架实现 MultiHeadSelfAttention、position-wise FFN、TransformerEncoderLayer 和 TransformerEncoder；可选实现简单 sinusoidal positional encoding。
2. public API 尽量参考 MyTorch-1 的命名和 shape 约定，但内部必须使用当前 TensorOp/autograd 与 V1-V4 后端。
3. 支持 batch-first 输入、attention mask（至少 padding 或 causal 其中一个，文档准确说明）、dropout、pre-norm 或 post-norm（明确选择）。
4. 复用 V4 Softmax/LayerNorm 的 auto dispatch；不另建重复实现。
5. 测试 shape、mask、deterministic eval、参数递归、CPU/CuPy forward/backward，以及极小序列 overfit smoke test。
6. 新增 examples/transformer_encoder_demo.py，展示随机 token/特征的小任务，不下载大模型或语料。

验收重点是完整可训练的 Encoder 组件，而不是大语言模型能力。不要扩展到 Decoder/GPT，也不要 commit/push，停止。
```

## V10：单机多卡数据并行与可选混合精度

```text
执行迁移 V10：实现最小单机多卡数据并行，不复制完整 Accelerator。只完成本版。

先审计 MyTorch-1 的 distributed/accelerate 设计，只借鉴接口。特别检查 all_reduce 的输出语义：不要对 param.grad/world_size 的两个临时数组调用 all_reduce 后丢弃结果；归约结果必须明确写回 param.grad，并且只除 world_size 一次。

任务：
1. 使用 cupyx.distributed/NCCLBackend 或等价 CuPy NCCL 能力，实现 init_process_group、rank/world_size/local_rank、barrier、broadcast、all_reduce、destroy。
2. 提供单机 launcher：一进程一 GPU；错误时清理子进程和 communicator。
3. 实现最小 DistributedDataParallel：初始参数广播、backward 后梯度 all-reduce average、optimizer step 前同步；不实现模型并行或多机网络层。
4. 实现 DistributedSampler，确保各 rank 样本不重叠、epoch seed 可控、尾批策略明确。
5. 可选实现轻量 GradScaler/FP16 路径；若数值稳定性未完成，宁可单独标 experimental，不得影响 FP32 DDP 验收。
6. 单元测试用 fake communicator 验证调用顺序和梯度写回；真实集成测试用 2 个 CUDA 进程比较单卡大 batch 与双卡等效梯度/参数更新。
7. 增加 examples/ddp_train.py 和故障排查文档，明确仅支持单机 NVIDIA 多卡。

没有至少 2 张 NVIDIA GPU 时，mock 只算单测通过，真实 DDP 必须列为未验证 blocker。不要声称 M1 或单张 4060 已验证多卡。不要 commit/push，停止。
```

## V11：训练监控后端与 Vue 面板

```text
执行迁移 V11：实现最小可用训练监控系统，对齐答辩中展示的训练可视化，但不把 dashboard 代码耦合进 autograd 核心。只完成本版。

任务：
1. 在训练端定义轻量 callback/logger 协议，将 epoch/step、train/val loss、accuracy、steer loss、throttle loss、learning rate、耗时、CPU 内存、GPU 显存写入 JSONL；无 dashboard 时训练照常运行。
2. dashboard/backend 使用 FastAPI 提供 run 列表、指标时间序列、最新状态、验证结果、数据摘要和 Grad-CAM 图片索引；需要实时更新时使用简单 WebSocket 或轮询。
3. dashboard/frontend 使用 Vue 做单页界面，至少包含：训练监控、loss/accuracy 曲线、steer/throttle loss、学习率/耗时/资源、验证结果、数据集统计、Grad-CAM 查看。
4. 不把图片或大日志塞入 Git；提供 demo log 生成器，用小型合成数据即可预览所有页面。
5. API 输入路径要限制在配置的 run 目录，避免任意文件读取；前后端端口、CORS 和路径可配置。
6. 测试 logger schema、API 主要端点和缺失/损坏日志处理；前端至少通过 lint/build，并增加最小组件测试或端到端 smoke。
7. README 给出同时启动训练、后端、前端的命令和截图生成方法。

完成后报告后端测试、前端 build 和实际打开页面的验证结果。不要顺手部署公网，不要 commit/push，停止。
```

## V12：Docker、可复现基准与答辩功能闭环

```text
执行迁移 V12：完成 GPU Docker、统一 benchmark、文档与答辩功能矩阵。只完成本版，不再扩张功能范围。

任务：
1. 提供 CPU Dockerfile 与 NVIDIA CUDA Dockerfile/compose profile；锁定兼容的 Python、CUDA、CuPy、Triton 版本，使用非 root 用户和合理 layer cache。GPU 容器运行依赖 NVIDIA Container Toolkit，文档写清。
2. 容器内至少能执行 import、CPU pytest、CUDA smoke、选定 fused-op benchmark 和 AutoDrive synthetic training；dashboard 可用独立 compose service。
3. 新增统一 benchmarks/run_all.py 或 shell-safe 等价入口，收集环境、git commit、设备、依赖版本、配置、预热、重复次数、正确性校验、延迟、吞吐和峰值显存，结果写 JSON/CSV。
4. 覆盖：CPU vs CuPy、Conv naive vs im2col vs Triton、四个融合算子 eager vs Triton、同步 vs 异步 DataLoader、AutoDrive 一小段训练；DDP 只有真实多卡时才出结果。
5. 新增 docs/PPT_FEATURE_MATRIX.md，将答辩每项内容映射到实现文件、测试、运行命令和最新实测结果：CuPy、算子融合、im2col、Triton Conv、异步 DataLoader、ResNet AutoDrive、双输出/动态油门、多地图、Grad-CAM、Transformer、并行训练、训练可视化、Docker。
6. 明确标注三种状态：implemented+tested、implemented+hardware-unverified、planned/unsupported。不得把 skipped test 写成 passed，不得沿用旧 PPT 数字冒充本仓库复现结果。
7. 更新根 README：架构图、安装矩阵、quickstart、支持矩阵、AutoDrive 流程、benchmark 方法、限制、许可证/引用。
8. 做最终质量检查：CPU-only 安装、CUDA extras 安装、pytest、lint/format、前端 build、Docker build；修复本版引入的问题，不做无关重构。

完成后给出最终验收报告和仍需真实硬件/模拟器完成的清单。不要自动 commit/push，停止。
```

## 每版复核命令（你可以在 Codex 完成后再发）

```text
现在不要继续下一版。请只对刚完成的版本做验收复核：
1. 展示 git status --short 与 git diff --stat；
2. 按任务书逐项列出“通过 / 未通过 / 未验证”，每项引用具体测试或命令；
3. 运行当前环境能运行的完整测试，不得把 skipped/xfailed 当作 passed；
4. 搜索是否意外引入 torch、硬编码 numpy、隐式 .get()/asnumpy、sys.path hack、大文件、__pycache__；
5. 指出最可能的三个回归风险；
6. 若有失败，只修复本版本范围内的问题并重新测试；
7. 不 commit、不 push、不开始下一版。
```

## 范围边界

本路线明确不迁移：GPT/nanochat、Tokenizer、FlashAttention、GQA/KV cache、MoE、大模型训练脚本、完整 Hugging Face 风格生态、完整多机 Accelerator。它们会稀释“简单教学框架 + CNN 优化 + AutoDrive”的主线，也不是当前答辩承诺的最低闭环。
