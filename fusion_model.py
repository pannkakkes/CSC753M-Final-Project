# Shared

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):
    def __init__(self, text_feats, audio_feats, video_feats, labels):
        self.t = torch.tensor(text_feats, dtype=torch.float32)
        self.a = torch.tensor(audio_feats, dtype=torch.float32)
        self.v = torch.tensor(video_feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.t[idx], self.a[idx], self.v[idx], self.y[idx]

# Multimodal fusion model with cross attention
class CrossAttentionFusion(nn.Module):
    def __init__(self, t_in, a_in, v_in,
                 embed_dim=128, num_heads=4, num_classes=4,
                 anchor="text", dropout=0.4):
        super().__init__()
        assert anchor in ("text", "audio", "video", "avg")
        self.anchor = anchor

        self.proj_t = nn.Linear(t_in, embed_dim)
        self.proj_a = nn.Linear(a_in, embed_dim)
        if v_in > 1000:
            self.proj_v = nn.Sequential(
                nn.Linear(v_in, 512), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(512, embed_dim)
            )
        else:
            self.proj_v = nn.Linear(v_in, embed_dim)

        self.avg_weights = nn.Parameter(torch.ones(3) / 3)
        self.attn_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.gate_layer = nn.Linear(embed_dim * 3, 3)

    def forward(self, t, a, v, return_attn=False):
        t_feat = self.proj_t(t).unsqueeze(1)
        a_feat = self.proj_a(a).unsqueeze(1)
        v_feat = self.proj_v(v).unsqueeze(1)

        if self.anchor == "text":
            q, kv1, kv2 = t_feat, a_feat, v_feat
        elif self.anchor == "audio":
            q, kv1, kv2 = a_feat, t_feat, v_feat
        elif self.anchor == "video":
            q, kv1, kv2 = v_feat, t_feat, a_feat
        else:  # avg
            w = torch.softmax(self.avg_weights, dim=0)
            q = w[0] * t_feat + w[1] * a_feat + w[2] * v_feat
            kv1, kv2 = t_feat, a_feat

        attn_out1, weights_1 = self.attn_1(q, kv1, kv1)
        attn_out2, weights_2 = self.attn_2(q, kv2, kv2)

        q_sq = q.squeeze(1)
        ctx1 = self.ln_1(attn_out1 + q).squeeze(1)
        ctx2 = self.ln_2(attn_out2 + q).squeeze(1)

        combined = torch.cat([q_sq, ctx1, ctx2], dim=1)
        gate_weights = torch.softmax(self.gate_layer(combined), dim=1)

        fused = torch.cat([
            gate_weights[:, 0:1] * q_sq,
            gate_weights[:, 1:2] * ctx1,
            gate_weights[:, 2:3] * ctx2,
        ], dim=1)

        logits = self.classifier(fused)
        if return_attn:
            return logits, gate_weights, weights_1, weights_2
        return logits