"""Matched frozen-backbone feature-modulation comparator for Paper 3.

This is an independently implemented *feature-modulation-inspired* control,
NOT a reproduction of any specific published method. The original reference
weights remain frozen. It supports explicit RGB/pose availability and must be
trained and evaluated on the same splits as the specialist router.
"""
import torch
from torch import nn
from .core import TemporalFusion


class FrozenFeatureModulation(nn.Module):
    """Small FiLM correction on frozen 128-d modality embeddings.

    Inputs: RGB [B,16,512], pose [B,16,13,2], availability [B,2].
    Modulation is conditioned on a two-bit observed availability mask.
    A frozen clean reference provides an anchored clean-logit path.
    """

    def __init__(self, reference_state, hidden=32):
        super().__init__()
        self.reference = TemporalFusion()
        self.reference.load_state_dict(reference_state, strict=True)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.conditioner = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, 512))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        self.head = nn.Linear(256, 15)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, rgb, pose, availability):
        batch = rgb.shape[0]
        if availability.shape != (batch, 2):
            raise ValueError("availability must have shape [B,2]")
        availability = availability.bool()
        if not availability.any(dim=1).all():
            raise ValueError("at least one modality must be available")
        mask = availability.float()
        rgb = rgb * mask[:, 0, None, None]
        pose = pose * mask[:, 1, None, None, None]
        with torch.no_grad():
            original = self.reference(rgb, pose)
            r = self.reference.rgb_encoder(rgb)
            p = self.reference.pose_encoder(pose.flatten(2))
        scale_shift = self.conditioner(mask)
        gamma_r, beta_r, gamma_p, beta_p = scale_shift.chunk(4, dim=-1)
        r = (r * (1 + gamma_r) + beta_r) * mask[:, 0, None]
        p = (p * (1 + gamma_p) + beta_p) * mask[:, 1, None]
        correction = self.head(torch.cat([r, p], dim=-1))
        return original + correction
