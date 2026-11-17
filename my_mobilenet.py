# my_mobilenet.py
import torch
import torch.nn as nn

# 1. 深度可分离卷积 (核心：降低计算量)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        # 深度卷积 (每个通道独立卷积)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU6(inplace=True)
        
        # SE 注意力模块 (通道注意力)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, in_ch // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // 4, in_ch, 1),
                nn.Hardsigmoid(inplace=True)
            )
            
        # 逐点卷积 (1x1 改变通道数)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        if self.use_se:
            x = x * self.se(x)
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x

# 2. 倒残差结构 (Inverted Residual Block)
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=False):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True)
            ])
            
        layers.append(DepthwiseSeparableConv(hidden_ch, out_ch, stride, use_se))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

# 3. 完整的自定义 MobileNetV3 Small
class MyMobileNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 第一层普通卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        
        # 堆叠倒残差块 (输入通道, 输出通道, 步长, 扩展倍率, 是否使用SE注意力)
        self.blocks = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=1, use_se=True),
            InvertedResidual(16, 24, stride=2, expand_ratio=4, use_se=False),
            InvertedResidual(24, 24, stride=1, expand_ratio=4, use_se=False),
            InvertedResidual(24, 40, stride=2, expand_ratio=4, use_se=True),
            InvertedResidual(40, 40, stride=1, expand_ratio=4, use_se=True),
            InvertedResidual(40, 80, stride=2, expand_ratio=4, use_se=False),
        )
        
        # 最后的 1x1 卷积和池化
        self.head = nn.Sequential(
            nn.Conv2d(80, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)