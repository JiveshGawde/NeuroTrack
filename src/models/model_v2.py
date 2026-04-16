import torch
import torch.nn as nn
import torchvision as tv


class NeuroTrackNN(nn.Module):

    def __init__(self, window_size: int = 3, num_heads: int = 8, d: int = 1408, num_classes: int = 4):
        super().__init__()
        efficient_net = tv.models.efficientnet_b2(weigths="IMAGENET1K_V1")

        self.backbone = nn.Sequential(
            efficient_net.features, efficient_net.avgpool)

        self.pos_emb = nn.Embedding(128, d)

        self.attn = nn.MultiheadAttention(num_heads=num_heads,embed_dim=d,batch_first=True)

        self.norm = nn.LayerNorm(d)

        self.head = nn.Sequential(
                nn.LayerNorm(d*2),
                nn.Linear(d*2, 256),
                nn.GELU(),
                nn.Linear(256, max(num_classes - 1, 1))
                )

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_n_blocks(self, n: int = 3):
        blocks = list(self.backbone[0].children())
        for block in blocks[-n:]:
            for b in block.parameters():
                b.requires_grad = True


    def encode_windows(self, windows: torch.Tensor):
        B, N, W, C, H, Ht = windows.shape

        x = windows.view(B*N*W, C, H, Ht)

        x = self.backbone(x).flatten(1)

        x = x.view(B * N, W, -1).mean(dim=1)

        return x.view(B, N, -1)

    def forward(self, X: torch.Tensor, mask=None):

        feats = self.encode_windows(X)

        pos = torch.arange(feats.size(1), device=feats.device)

        feats = feats + self.pos_emb(pos)

        attn_out, _ = self.attn(feats, feats, feats, key_padding_mask=mask)

        attn_out = self.norm(feats + attn_out).mean(dim=1)

        if mask is not None:
            feats = feats.masked_fill(mask.unsqueeze(-1), float('-inf'))

        pool_out, _ = feats.max(dim=1)

        return self.head(torch.cat([attn_out, pool_out], dim=-1))


    def predict(self, logits):

        return (torch.sigmoid(logits) > 0.5).sum(dim=-1)
