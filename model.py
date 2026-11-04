"""Modelle: eingefrorenes Foundation-Backbone + trainierbarer leichter Conv-Head.

Neue Varianten (z. B. CEM500k-Backbone, ResNet18 from scratch) werden als Klasse mit
gleichem Interface (forward(x) -> Logits [B,1,H,W], trainable_parameters(), trainable_state_dict())
geschrieben und in MODEL_REGISTRY eingetragen. Auswahl über cfg["model"].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_act(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.GELU())


class LightConvHead(nn.Module):
    """Patch-Feature-Map [B,C,h,w] -> Logits [B,1,h*up,w*up].

    BN -> 1x1-Projektion -> 3x3-Conv -> 2x (Bilinear-Upsample x2 + 3x3-Conv) -> 1x1-Conv.
    Bewusst klein (~1-2 M Parameter bei C=1024, ch=256), damit das Backbone das Ergebnis dominiert.
    """

    def __init__(self, in_channels: int, channels: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.proj = _conv_bn_act(in_channels, channels, k=1)
        self.block0 = _conv_bn_act(channels, channels, k=3)
        self.up1 = _conv_bn_act(channels, channels // 2, k=3)
        self.up2 = _conv_bn_act(channels // 2, channels // 4, k=3)
        self.drop = nn.Dropout2d(dropout)
        self.out = nn.Conv2d(channels // 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block0(self.proj(self.norm(x)))
        x = self.drop(x)
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.out(x)


class DinoV3Segmenter(nn.Module):
    """Eingefrorenes DINOv3 (HuggingFace) + LightConvHead."""

    def __init__(self, backbone_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
                 n_feature_layers: int = 1, head_channels: int = 256, head_dropout: float = 0.1,
                 backbone_dtype_amp: bool = True):
        super().__init__()
        from transformers import AutoModel  # lazy import

        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        cfg = self.backbone.config
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.n_register = int(getattr(cfg, "num_register_tokens", 0))
        depth = int(cfg.num_hidden_layers)
        # hidden_states hat depth+1 Einträge (Index 0 = Embedding). Gleichmäßig verteilte Layer, letzter = depth.
        self.layer_ids = sorted({round(depth * (i + 1) / n_feature_layers) for i in range(n_feature_layers)})
        self.depth = depth
        self.head = LightConvHead(cfg.hidden_size * len(self.layer_ids), head_channels, head_dropout)

    def train(self, mode: bool = True):  # Backbone bleibt immer im eval-Modus
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Bildgröße {H}x{W} muss Vielfaches von {p} sein"
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        feats = []
        for i in self.layer_ids:
            h = out.last_hidden_state if i == self.depth else out.hidden_states[i]
            feats.append(h[:, 1 + self.n_register :, :])  # CLS + Register-Tokens entfernen
        f = torch.cat(feats, dim=-1)  # [B, N, C*L]
        h_, w_ = H // p, W // p
        assert f.shape[1] == h_ * w_, f"Token-Anzahl {f.shape[1]} != {h_}x{w_}"
        return f.transpose(1, 2).reshape(B, -1, h_, w_).float().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        logits = self.head(feats)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def trainable_parameters(self):
        return self.head.parameters()

    def trainable_state_dict(self):
        return self.head.state_dict()


MODEL_REGISTRY = {
    "dinov3": DinoV3Segmenter,
    # "cem500k": ..., "resnet18_scratch": ...  # später
}


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("model", "dinov3")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unbekanntes Modell '{name}'. Verfügbar: {list(MODEL_REGISTRY)}")
    if name == "dinov3":
        return DinoV3Segmenter(
            backbone_name=cfg["backbone_name"],
            n_feature_layers=int(cfg.get("n_feature_layers", 1)),
            head_channels=int(cfg.get("head_channels", 256)),
            head_dropout=float(cfg.get("head_dropout", 0.1)),
        )
    return MODEL_REGISTRY[name](**cfg.get("model_kwargs", {}))
