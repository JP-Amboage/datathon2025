import torch
import torch.nn as nn


# class ProfileAttentionClassifier(nn.Module):
#     def __init__(self, emb_dim=1024, num_heads=8):
#         super().__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=emb_dim,
#             nhead=num_heads,
#             dim_feedforward=2048,
#             dropout=0.1,
#             activation='gelu',
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Sequential(
#             nn.Linear(emb_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):  # x shape: (batch, 7, 1024)
#         x = self.encoder(x)  # shape still (batch, 7, 1024)
#         x = x.permute(0, 2, 1)  # for pooling: (batch, emb_dim, seq_len)
#         pooled = self.pool(x).squeeze(-1)  # (batch, emb_dim)
#         return self.classifier(pooled)



class AttentionPool(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, emb_dim))  # (1, emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):  # x: (batch, 7, emb_dim)
        # Dot-product attention with learnable query
        q = self.query.unsqueeze(0).repeat(x.size(0), 1, 1)  # (batch, 1, emb_dim)
        k = self.linear(x)  # (batch, 7, emb_dim)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)  # (batch, 1, 7)
        pooled = attn @ x  # (batch, 1, emb_dim)
        return pooled.squeeze(1)  # (batch, emb_dim)



class PositionalEmbedding(nn.Module):
    def __init__(self, num_positions, emb_dim):
        super().__init__()
        self.pe = nn.Embedding(num_positions, emb_dim)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)  # shape: (seq_len,)
        return x + self.pe(positions)


class MLPHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(emb_dim, 1)

    def forward(self, x):
        x = x + self.block(x)  # residual MLP
        return self.out(x)


class ProfileAttentionClassifier(nn.Module):
    def __init__(self, emb_dim=1024, num_heads=8):
        super().__init__()
        self.pos_encoder = PositionalEmbedding(num_positions=7, emb_dim=emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=4096,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.pool = AttentionPool(emb_dim)
        self.classifier = MLPHead(emb_dim)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        pooled = self.pool(x)
        return self.classifier(pooled)


