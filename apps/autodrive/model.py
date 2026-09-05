"""A small MyTorch ResNet with steering and bounded-throttle heads."""

import mytorch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, *, device=None,
                 dtype="float32"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1,
            bias=False, device=device, dtype=dtype,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False,
            device=device, dtype=dtype,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False,
                    device=device, dtype=dtype,
                ),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
            )

    def forward(self, inputs):
        residual = inputs if self.projection is None else self.projection(inputs)
        values = nn.ReLU()(self.bn1(self.conv1(inputs)))
        values = self.bn2(self.conv2(values))
        return nn.ReLU()(values + residual)


class AutoDriveResNet(nn.Module):
    """Shared lightweight ResNet backbone with two physical-output heads."""

    def __init__(self, base_channels=16, blocks=(1, 1, 1), *,
                 throttle_min=0.0, throttle_max=1.0, device=None,
                 dtype="float32"):
        super().__init__()
        base_channels = int(base_channels)
        blocks = tuple(int(value) for value in blocks)
        if base_channels <= 0 or len(blocks) != 3 or min(blocks) <= 0:
            raise ValueError("base_channels and all three block counts must be positive")
        if not throttle_min < throttle_max:
            raise ValueError("throttle_min must be smaller than throttle_max")
        self.base_channels = int(base_channels)
        self.blocks = blocks
        self.throttle_min = float(throttle_min)
        self.throttle_max = float(throttle_max)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False,
                      device=device, dtype=dtype),
            nn.BatchNorm2d(base_channels, device=device, dtype=dtype),
            nn.ReLU(),
        )
        self.stage1 = self._stage(
            base_channels, base_channels, blocks[0], 1, device, dtype
        )
        self.stage2 = self._stage(
            base_channels, base_channels * 2, blocks[1], 2, device, dtype
        )
        self.stage3 = self._stage(
            base_channels * 2, base_channels * 4, blocks[2], 2, device, dtype
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        features = base_channels * 4
        self.steering_head = nn.Linear(features, 1, device=device, dtype=dtype)
        self.throttle_head = nn.Linear(features, 1, device=device, dtype=dtype)

    @staticmethod
    def _stage(in_channels, out_channels, count, stride, device, dtype):
        modules = [BasicBlock(
            in_channels, out_channels, stride, device=device, dtype=dtype
        )]
        modules.extend(BasicBlock(
            out_channels, out_channels, device=device, dtype=dtype
        ) for _ in range(1, count))
        return nn.Sequential(*modules)

    def forward_features(self, inputs):
        values = self.stem(inputs)
        values = self.stage1(values)
        values = self.stage2(values)
        return self.stage3(values)

    def forward(self, inputs):
        return self.forward_heads(self.forward_features(inputs))

    def forward_heads(self, features):
        features = self.pool(features)
        features = features.reshape((features.shape[0], features.shape[1]))
        steering = nn.Tanh()(self.steering_head(features))
        throttle_unit = nn.Sigmoid()(self.throttle_head(features))
        throttle = self.throttle_min + (
            self.throttle_max - self.throttle_min
        ) * throttle_unit
        return steering, throttle
