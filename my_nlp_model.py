# my_nlp_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTextCNN(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, num_classes=3):
        super().__init__()
        # 词嵌入层：将整型 token 映射为浮点向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 三个不同尺寸的并行卷积核（模拟提取 2-gram, 3-gram, 4-gram 特征）
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 32, kernel_size=k) for k in (2, 3, 4)
        ])
        
        # 分类器
        self.fc = nn.Linear(32 * 3, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len) -> 例如 (1, 32)
        # 注意：NLP 模型通常不需要在 forward 里写 x.long()，
        # 因为输入本身就是通过 dtypes=[torch.long] 生成的整型
        x = self.embedding(x)                  # (B, seq_len, embed_dim)
        x = x.transpose(1, 2)                  # (B, embed_dim, seq_len)
        
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))                # (B, out_ch, L_out)
            # 全局最大池化，提取每个卷积核最显著的特征
            pooled.append(F.max_pool1d(c, c.size(2)).squeeze(2)) 
            
        x = torch.cat(pooled, dim=1)           # (B, 32*3)
        return self.fc(x)                      # (B, num_classes)