# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 原有 15 个模型（保持不变）
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.squeeze(-1))


class CNN2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SimpleResNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock2D(32)
        self.layer2 = ResidualBlock2D(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale


class CNN2D_ChannelAttention(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.attn = ChannelAttention(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CNN2D_SpatialAttention(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.attn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, out_channels, 1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = torch.cat([e1, d], dim=1)
        return self.dec(d)


class CNN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, state=None):
        if state is None:
            b, _, h, w = x.shape
            state = (
                torch.zeros(b, self.hidden_dim, h, w, device=x.device),
                torch.zeros(b, self.hidden_dim, h, w, device=x.device),
            )
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv(combined))
        i, f, o, g = gates.chunk(4, dim=1)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class TinyConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, num_classes=5):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        state = None
        for t in range(x.size(1)):
            _, state = self.cell(x[:, t], state)
        h = state[0]
        return self.fc(self.pool(h).flatten(1))


class SimpleRNN(nn.Module):
    def __init__(self, input_size=16, hidden=32, num_classes=5):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=16, hidden=32, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class BiRNN(nn.Module):
    def __init__(self, input_size=16, hidden=32, num_classes=5):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        _, h = self.rnn(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)


class BiLSTM(nn.Module):
    def __init__(self, input_size=16, hidden=32, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)


class LSTMAttention(nn.Module):
    def __init__(self, input_size=16, hidden=32, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.attn_score = nn.Linear(hidden, 1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        scores = self.attn_score(outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(context)


class TinyTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=32, nhead=4, num_layers=2, num_classes=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1])


# ============================================================
# 新增 CV 模型
# ============================================================

class VGG11Small(nn.Module):
    """极简 VGG11，用于图像识别，输入 (B, 3, 64, 64)。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# class ResNet18Small(nn.Module):
#     """极简 ResNet18 风格，输入 (B, 3, 64, 64)。"""
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=1, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
#         )
#         self.layer1 = self._make_layer(32, 32, 2)
#         self.layer2 = self._make_layer(32, 64, 2, stride=2)
#         self.layer3 = self._make_layer(64, 128, 2, stride=2)
#         self.layer4 = self._make_layer(128, 256, 2, stride=2)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(256, num_classes)

#     @staticmethod
#     def _make_layer(in_ch, out_ch, blocks, stride=1):
#         layers = []
#         downsample = None
#         if stride != 1 or in_ch != out_ch:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_ch),
#             )
#         layers.append(ResidualBlock2D(out_ch))  # 简化：复用已有残差块
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock2D(out_ch))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x); x = self.layer2(x)
#         x = self.layer3(x); x = self.layer4(x)
#         return self.fc(self.pool(x).flatten(1))


class BasicBlock(nn.Module):
    """标准 ResNet 残差块，包含通道/尺寸不匹配时的 1x1 捷径投影"""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18Small(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
        )
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).flatten(1))


class ViTTiny(nn.Module):
    """极简 ViT，输入 (B, 3, 32, 32)，patch_size=8。"""
    def __init__(self, num_classes=10, dim=64, depth=2, heads=4):
        super().__init__()
        patch_size = 8
        num_patches = (32 // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class YOLODetectionHead(nn.Module):
    """简化的 YOLO 风格检测头，输入 (B, 3, 128, 128)。"""
    def __init__(self, num_classes=5, in_ch=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.detect = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 + num_classes, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)           # (B,128,16,16)
        out = self.detect(feat)           # (B, 4+nc, 16,16)
        return out.flatten(2).transpose(1, 2)  # (B, 256, 4+nc)


class DeepLabV3Small(nn.Module):
    """简化 DeepLabV3 风格分割头，输入 (B, 3, 128, 128)。"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.aspp = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.aspp(feat)
        out = self.head(feat)
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)


class MaskRCNNLight(nn.Module):
    """极简实例分割头：backbone + 固定网格 RoI 池化 + mask head。"""
    def __init__(self, num_classes=5, mask_size=14):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        self.mask_size = mask_size

    # def forward(self, x):
    #     feat = self.backbone(x)           # (B,128,16,16)
    #     # 简化 RoI：把特征图分成 2x2 网格，各自自适应池化到 mask_size
    #     b, c, h, w = feat.shape
    #     rois = feat.view(b, c, 2, h // 2, 2, w // 2).permute(0, 2, 4, 1, 3, 5)
    #     rois = rois.reshape(b * 4, c, h // 2, w // 2)
    #     rois = F.adaptive_max_pool2d(rois, (self.mask_size, self.mask_size))
    #     masks = self.mask_head(rois)      # (B*4, nc, mask_size, mask_size)
    #     return masks.view(b, 4, -1, self.mask_size, self.mask_size)


    def forward(self, x):
        feat = self.backbone(x)           # (B,128,16,16)
        b, c, h, w = feat.shape
        rois = feat.view(b, c, 2, h // 2, 2, w // 2).permute(0, 2, 4, 1, 3, 5)
        rois = rois.reshape(b * 4, c, h // 2, w // 2)
        rois = F.max_pool2d(rois, kernel_size=2)   # 替代 adaptive_max_pool2d
        masks = self.mask_head(rois)
        return masks.view(b, 4, -1, masks.size(2), masks.size(3))

class ImageCaptioning(nn.Module):
    """极简图像描述：CNN encoder + LSTM decoder，输入 (B, 3, 64, 64)。"""
    def __init__(self, vocab_size=100, embed_dim=64, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(64, hidden)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        ctx = self.fc(self.encoder(x)).unsqueeze(1)       # (B,1,H)
        seq_len = 5
        tokens = torch.zeros(x.size(0), seq_len, dtype=torch.long, device=x.device)
        emb = self.embed(tokens)                          # (B,T,E)
        ctx = ctx.expand(-1, seq_len, -1)
        out, _ = self.lstm(torch.cat([emb, ctx], dim=-1))
        return self.out(out)


class DCGANGenerator(nn.Module):
    """文生图生成器（DCGAN 风格），输入 (B, 100)。"""
    def __init__(self, z_dim=100, img_size=32, base=64):
        super().__init__()
        self.init_size = img_size // 4
        self.fc = nn.Linear(z_dim, base * self.init_size * self.init_size)
        self.net = nn.Sequential(
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, base // 2, 4, 2, 1),
            nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base // 2, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, self.init_size, self.init_size)
        return self.net(x)


# ============================================================
# 新增 NLP 模型
# ============================================================

class BERTClassifier(nn.Module):
    """极简 BERT 文本分类，输入 (B, 32) 整数序列。"""
    def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 32, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.long()
        x = self.embed(x) + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return self.fc(self.norm(x[:, 0]))


# class GPT2Small(nn.Module):
#     """极简 GPT 文本生成，输入 (B, 32) 整数序列。"""
#     def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, d_model)
#         self.pos = nn.Parameter(torch.zeros(1, 32, d_model))
#         decoder_layer = nn.TransformerEncoderLayer(  # 用 EncoderLayer 模拟因果注意力，简化
#             d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
#         )
#         self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
#         self.norm = nn.LayerNorm(d_model)
#         self.head = nn.Linear(d_model, vocab_size)

#     def forward(self, x):
#         x = x.long()
#         x = self.embed(x) + self.pos[:, : x.size(1)]
#         mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
#         x = self.decoder(x, mask=mask)
#         return self.head(self.norm(x))


# class GPT2Small(nn.Module):
#     def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, d_model)
#         self.pos = nn.Parameter(torch.zeros(1, 32, d_model))
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=d_model, nhead=nhead,
#                 dim_feedforward=d_model * 4, batch_first=True,
#             )
#             for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(d_model)
#         self.head = nn.Linear(d_model, vocab_size)

#     def forward(self, x):
#         x = x.long()
#         x = self.embed(x) + self.pos[:, : x.size(1)]
#         # 直接传入 is_causal=True，避免 generate_square_subsequent_mask 的 .item() 检测
#         for layer in self.layers:
#             x = layer(x, src_mask=None, is_causal=True)
#         return self.head(self.norm(x))


# class Seq2SeqTranslator(nn.Module):
#     """极简 Encoder-Decoder 机器翻译，输入 (B, 20) 整数序列。"""
#     def __init__(self, vocab_size=1000, embed_dim=64, hidden=64):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         self.encoder = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
#         self.decoder = nn.LSTM(embed_dim + hidden * 2, hidden, batch_first=True)
#         self.out = nn.Linear(hidden, vocab_size)

#     def forward(self, x):
#         x = x.long()
#         emb = self.embed(x)
#         _, (h, c) = self.encoder(emb)
#         h = torch.cat([h[0], h[1]], dim=1).unsqueeze(1)  # (B,1,2H)
#         c = c[0].unsqueeze(1)
#         tgt_len = 5
#         tgt = torch.zeros(x.size(0), tgt_len, dtype=torch.long, device=x.device)
#         dec_in = self.embed(tgt)
#         ctx = h.expand(-1, tgt_len, -1)
#         out, _ = self.decoder(torch.cat([dec_in, ctx], dim=-1), (h, c))
#         return self.out(out)


class GPT2Small(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 32, d_model))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.long()
        # 1. 生成一个明确的、与序列长度匹配的因果掩码
        seq_len = x.size(1)
        # torch.triu 会生成一个上三角矩阵，与因果掩码的需求一致
        # 这种写法对 ONNX 导出更友好
        causal_mask = torch.triu(
            # torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=torch.float32),
            torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=self.pos.dtype),
            diagonal=1
        )
        
        x = self.embed(x) + self.pos[:, :seq_len]
        
        # 2. 将明确的掩码传递给每一层
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        return self.head(self.norm(x))


    # def forward(self, x):
    #     x = x.long()
    #     seq_len = x.size(1)
        
    #     # ✅ 修复：使用 torch.bool 类型的掩码，避开 -inf 的数值转换 Bug
    #     causal_mask = torch.triu(
    #         torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
    #         diagonal=1
    #     )
        
    #     x = self.embed(x) + self.pos[:, :seq_len]
        
    #     for layer in self.layers:
    #         x = layer(x, src_mask=causal_mask)
            
    #     return self.head(self.norm(x))


# class Seq2SeqTranslator(nn.Module):
#     def __init__(self, vocab_size=1000, embed_dim=64, hidden=64):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         self.encoder = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
#         self.decoder = nn.LSTM(embed_dim + hidden * 2, hidden, batch_first=True)
#         self.out = nn.Linear(hidden, vocab_size)

#     def forward(self, x):
#         x = x.long()
#         emb = self.embed(x)                            # (B, T, E)
#         _, (h, c) = self.encoder(emb)                   # h: (2, B, H)

#         # 显式取正向与反向的最终隐藏状态，保证形状为 (B, 1, 2H)
#         h_fwd = h[0].unsqueeze(1)                       # (B, 1, H)
#         h_bwd = h[1].unsqueeze(1)                       # (B, 1, H)
#         h_cat = torch.cat([h_fwd, h_bwd], dim=-1)       # (B, 1, 2H)

#         # 初始化解码器的 cell state 为全零，避免维度推断错误
#         c_init = torch.zeros_like(h_cat[:, :1, :self.decoder.hidden_size])  # (B, 1, H)

#         tgt_len = 5
#         tgt = torch.zeros(x.size(0), tgt_len, dtype=torch.long, device=x.device)
#         dec_in = self.embed(tgt)                         # (B, T, E)

#         # 将编码器上下文沿时间步复制，并与解码器输入拼接
#         ctx = h_cat.expand(-1, tgt_len, -1)              # (B, T, 2H)
#         dec_input = torch.cat([dec_in, ctx], dim=-1)     # (B, T, E+2H)

#         out, _ = self.decoder(dec_input, (h_cat, c_init))
#         return self.out(out)


class Seq2SeqTranslator(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Encoder 是双向的
        self.encoder = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        
        # 新增桥接层：将双向的 2*hidden 降维到 Decoder 需要的 hidden
        self.bridge_h = nn.Linear(hidden * 2, hidden)
        self.bridge_c = nn.Linear(hidden * 2, hidden)
        
        # Decoder 的输入特征维度是 embed_dim + 双向上下文(2*hidden)
        self.decoder = nn.LSTM(embed_dim + hidden * 2, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = x.long()
        emb = self.embed(x)                            # (B, T, E)
        _, (h, c) = self.encoder(emb)                   # h, c: (2, B, H)

        # 1. 拼接正向和反向的最终状态
        h_cat = torch.cat([h[0], h[1]], dim=-1)         # (B, 2H)
        c_cat = torch.cat([c[0], c[1]], dim=-1)         # (B, 2H)
        
        # 2. 通过桥接层降维，并增加 num_layers 维度 (1, B, H)
        h_init = torch.tanh(self.bridge_h(h_cat)).unsqueeze(0)  
        c_init = torch.tanh(self.bridge_c(c_cat)).unsqueeze(0)  

        tgt_len = 5
        tgt = torch.zeros(x.size(0), tgt_len, dtype=torch.long, device=x.device)
        dec_in = self.embed(tgt)                         # (B, T, E)

        # 3. 将编码器上下文沿时间步复制，并与解码器输入拼接
        ctx = h_cat.unsqueeze(1).expand(-1, tgt_len, -1) # (B, T, 2H)
        dec_input = torch.cat([dec_in, ctx], dim=-1)     # (B, T, E+2H)

        # 4. 将映射后的 (1, B, H) 传入 Decoder
        out, _ = self.decoder(dec_input, (h_init, c_init))
        return self.out(out)


class TextCNN(nn.Module):
    """TextCNN 文本分类，输入 (B, 32) 整数序列。"""
    def __init__(self, vocab_size=1000, embed_dim=64, num_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 32, k) for k in (2, 3, 4)
        ])
        self.fc = nn.Linear(32 * 3, num_classes)

    def forward(self, x):
        x = x.long()
        x = self.embed(x).transpose(1, 2)          # (B,E,T)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            pooled.append(F.max_pool1d(c, kernel_size=c.size(2)).squeeze(-1))  # 替代 adaptive_max_pool1d
        return self.fc(torch.cat(pooled, dim=1))

    # def forward(self, x):
    #     x = x.long()
    #     x = self.embed(x).transpose(1, 2)          # (B,E,T)
    #     pooled = []
    #     for conv in self.convs:
    #         c = F.relu(conv(x))
    #         pooled.append(F.adaptive_max_pool1d(c, 1).squeeze(-1))
    #     return self.fc(torch.cat(pooled, dim=1))


class BiGRUSequenceLabeler(nn.Module):
    """双向 GRU 序列标注，输入 (B, 20, 16)。"""
    def __init__(self, input_dim=16, hidden=32, num_labels=5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_labels)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)