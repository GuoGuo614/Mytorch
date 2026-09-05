# MyTorch 自动驾驶使用说明

`apps/autodrive` 是基于 MyTorch 的自动驾驶应用，包含旧数据转换、模拟器
数据采集、数据检查、模型训练、断点续训、评估、闭环自动驾驶和 Grad-CAM。

模型使用轻量 ResNet 主干，同时输出：

- `steering`：转向，范围为 `[-1, 1]`；
- `throttle`：油门，默认范围为 `[0, 1]`。

`apps/auto-drive(paddle)` 仅作为 Paddle 参考实现，不参与 MyTorch 训练和推理。

## 1. 环境安装

## 2. 支持的地图

| 地图参数 | DonkeyCar 环境 |
|---|---|
| `generated-track` | `donkey-generated-track-v0` |
| `mountain-track` | `donkey-mountain-track-v0` |
| `warren-track` | `donkey-warren-track-v0` |
| `warehouse` | `donkey-warehouse-v0` |
| `circuit` | `donkey-circuit-launch-track-v0` |

通常只需要指定 `--map`。如果模拟器使用了自定义环境 ID，可以额外传入
`--env-name`，但必须保证它和 `--map` 表示同一张地图。

## 3. 采集新数据

先启动 DonkeyCar 模拟器并选择对应赛道。以 Warren 地图为例：

```powershell
python -m apps.autodrive collect `
  --map warren-track `
  --output data/DonkeyCar/collected/warren-track `
  --max-samples 10000
```

建议每张地图使用独立的 `--output`。`--max-samples` 会递归统计该输出目录
已有的 PNG，因此上面的命令表示 Warren 地图累计达到 10,000 张后停止：

- 已有 0 张时最多再采集 10,000 张；
- 已有 8,700 张时最多再采集 1,300 张；
- 已达到 10,000 张时直接退出。

如需限制本次启动最多采集多少张，可增加单次上限：

```powershell
python -m apps.autodrive collect `
  --map warren-track `
  --output data/DonkeyCar/collected/warren-track `
  --max-samples 10000 `
  --max-steps 2000
```

采集窗口必须获得键盘焦点：

| 按键 | 功能 |
|---|---|
| `A` | 向左转 |
| `D` | 向右转 |
| `W` | 逐渐增加油门 |
| 松开 `W` | 油门逐渐回落到基础油门 |
| 松开 `A/D` | 方向逐渐回正 |
| `Space` | 刹车，油门降为 0 |
| `Esc` | 提前结束采集 |

常用控制参数：

```powershell
python -m apps.autodrive collect `
  --map mountain-track `
  --output data/DonkeyCar/collected/mountain-track `
  --max-samples 10000 `
  --base-throttle 0.2 `
  --max-throttle 0.5 `
  --throttle-rise 0.2 `
  --throttle-decay 0.1 `
  --steering-rate 2.0 `
  --steering-return 3.0 `
  --steering-limit 1.0 `
  --fps 20
```

所有控制参数都会写入本次 run 的 `metadata.json`。

### 采集结果命名

每次模拟器 episode 使用一个独立 UUID。目录名称包含地图名：

```text
data/DonkeyCar/collected/warren-track/
└── warren-track_82ecae1b1ef44b888d91be37d6344c4e/
    ├── metadata.json
    ├── records.jsonl
    ├── 00000000.png
    ├── 00000001.png
    └── ...
```

图片在每个 run 内从 `00000000.png` 开始编号。文件名不保存标签，方向、
油门、地图、时间戳、run ID 和帧编号统一记录在 `records.jsonl` 中。

一次采集过程中如果模拟器结束并重置 episode，会产生新的 UUID 目录。

### 删除一次有问题的采集

先按照修改时间查看最近的 run：

```powershell
Get-ChildItem "data/DonkeyCar/collected/warren-track" -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object Name, LastWriteTime
```

确认 UUID 后删除准确目录：

```powershell
Remove-Item -LiteralPath "C:\code\Mytorch\data\DonkeyCar\collected\warren-track\warren-track_具体UUID" -Recurse -Force
```

如果已经生成训练 manifest，删除原始 run 后必须重新执行数据检查和导出。

## 4. 检查数据并生成训练 Manifest

新采集的数据先执行检查，再按完整 run 划分训练集和验证集：

```powershell
python -m apps.autodrive audit `
  --collection-root data/DonkeyCar/collected `
  --output data/DonkeyCar/collected_manifest.jsonl `
  --val-ratio 0.2 `
  --seed 256
```

检查内容包括：

- 无法读取或缺失的图片；
- 缺失、非有限或超出物理范围的标签；
- 解码后内容相同的重复图片；
- 每张地图的样本数量；
- steering/throttle 的范围、均值、标准差和直方图；
- 同一个 run 是否同时进入训练集和验证集。

存在坏数据或重复图片时不会导出 manifest。每张地图至少需要两个独立
run，程序才可以保证训练集和验证集都包含该地图。划分比例按 run 数量计算，
所以当各 run 长度不同时，图片数量不一定严格为 8:2。

只检查已有 manifest，不重新导出：

```powershell
python -m apps.autodrive audit `
  --manifest data/DonkeyCar/collected_manifest.jsonl
```

## 5. 转换旧版文件名数据

旧数据的标签保存在图片文件名中，需要转换一次：

```powershell
python -m apps.autodrive manifest `
  --data-root data/DonkeyCar `
  --output data/DonkeyCar/manifest.jsonl `
  --seed 256 `
  --default-throttle 0.2 `
  --group-size 500
```

不指定 `--maps` 时会转换所有检测到的地图。只转换部分地图：

```powershell
python -m apps.autodrive manifest `
  --data-root data/DonkeyCar `
  --output data/DonkeyCar/mountain_manifest.jsonl `
  --maps mountain-track
```

旧目录已经丢失真实采集 run 边界，因此转换器使用连续帧号块作为伪 run。
`data_circuit` 没有油门标签时使用 `--default-throttle 0.2`。

## 6. 训练模型

### 训练单张地图

```powershell
python -m apps.autodrive train `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --maps warren-track `
  --device cuda `
  --epochs 10 `
  --batch-size 32 `
  --num-workers 2
```

没有指定 `--checkpoint` 时，会自动把地图名称加入权重文件名：

```text
checkpoints/autodrive_warren-track.npz
checkpoints/autodrive_warren-track.json
```

NPZ 保存模型参数、BatchNorm 状态、Adam 状态、epoch 和训练配置；JSON 保存
推理需要的模型结构、归一化、控制范围和对应 NPZ 路径。


省略 `--maps` 表示使用 manifest 中的所有地图。也可以手动指定输出名称：

```powershell
python -m apps.autodrive train `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --maps warren-track `
  --device cuda `
  --checkpoint checkpoints/my_warren_model.npz
```

### CPU 训练

```powershell
python -m apps.autodrive train `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --maps warren-track `
  --device cpu `
  --epochs 10
```

每个 epoch 输出一行 JSON，包含总 loss、steering loss、throttle loss、学习率、
耗时、样本数和验证集 MAE。训练损失为：

```text
steering_mse + lambda_throttle * throttle_mse
```

使用 `--lambda-throttle` 调整油门损失权重。

## 7. 断点续训

使用和原训练相同的 manifest、地图及模型参数，并增加 `--resume`：

```powershell
python -m apps.autodrive train `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --maps warren-track `
  --device cuda `
  --epochs 20 `
  --resume
```

`--epochs 20` 表示训练到第 20 个 epoch，不是额外训练 20 个 epoch。默认文件名
仍然解析为 `checkpoints/autodrive_warren-track.npz`。如果首次训练手动指定了
`--checkpoint`，续训时也必须传入同一路径。

## 8. 评估模型

评估 Warren 模型：

```powershell
python -m apps.autodrive evaluate `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --maps warren-track `
  --device cuda
```

程序默认加载 `checkpoints/autodrive_warren-track.npz`，输出验证集总损失、两个
子损失以及 steering/throttle MAE。评估自定义权重：

```powershell
python -m apps.autodrive evaluate `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --checkpoint checkpoints/my_warren_model.npz `
  --device cuda
```

## 9. 启动自动驾驶

先启动 DonkeyCar 模拟器并选择与模型对应的地图，然后运行：

```powershell
python -m apps.autodrive drive `
  --config checkpoints/autodrive_warren-track.json `
  --device cuda `
  --map warren-track `
  --max-steps 6000 `
  --log-interval 50
```

程序从 JSON 找到配套 NPZ，使用保存的图像尺寸和归一化配置处理摄像头帧，
然后同时预测 steering 和 throttle，并执行范围裁剪和平滑。

如果训练集具有真实油门标签，会使用模型预测油门；如果整个训练集只有旧版
默认油门标签，JSON 会记录固定油门模式，自动驾驶时使用 0.2。

推理异常时根据 JSON 的 `failure_mode` 刹停或使用安全固定油门；连续失败达到
上限后结束驾驶并关闭模拟器环境。

## 10. 生成 Grad-CAM

对一张 Warren 图片生成 steering Grad-CAM：

```powershell
python -m apps.autodrive gradcam `
  --config checkpoints/autodrive_warren-track.json `
  --image data/DonkeyCar/collected/warren-track/warren-track_具体UUID/00000000.png `
  --output runs/warren_steering_cam.png `
  --head steering `
  --device cuda
```

查看油门 head：

```powershell
python -m apps.autodrive gradcam `
  --config checkpoints/autodrive_warren-track.json `
  --image data/DonkeyCar/collected/warren-track/warren-track_具体UUID/00000000.png `
  --output runs/warren_throttle_cam.png `
  --head throttle `
  --device cuda
```

输出图片保持原始尺寸。红色表示对当前输出有较高正向贡献，蓝色表示归一化后
响应较低。全零热力图也是合法结果，不代表程序运行失败。

## 11. 运行测试

运行全部测试：

```powershell
python -m pytest -q
```

只测试自动驾驶 V8：

```powershell
python -m pytest tests/test_autodrive_v8.py -q
```

自动化测试覆盖动态油门、采集文件、累计图片上限、manifest、控制裁剪、异常
处理和 CPU/CUDA Grad-CAM。真实模拟器键盘交互、断连行为、赛道完成率和圈速
