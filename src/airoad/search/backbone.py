from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50


def _pick_device(user: Optional[str] = None) -> torch.device:
    if user:
        if user == "cpu":
            return torch.device("cpu")
        if user == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if (user.isdigit() or user == "cuda") and torch.cuda.is_available():
            return torch.device("cuda:0" if user == "cuda" else f"cuda:{user}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class SearchBackboneConfig:
    name: str = "auto"  # "auto" | "open-clip:ViT-B-32" | "resnet50"
    device: Optional[str] = None
    dtype: str = "float32"  # "float32" | "float16" (fp16 disabled on CPU)
    image_size: int = 224


class ImageTextBackbone(nn.Module):
    """Image+Text backbone (OpenCLIP) with image-only fallback (ResNet-50)."""

    def __init__(self, cfg: SearchBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.dtype = (
            torch.float16
            if (cfg.dtype == "float16" and self.device.type != "cpu")
            else torch.float32
        )
        self._has_text = False

        if cfg.name == "auto" or cfg.name.startswith("open-clip"):
            try:
                import open_clip

                name = cfg.name.split("open-clip:", 1)[1] if ":" in cfg.name else "ViT-B-32"
                self.model, _, self.transform = open_clip.create_model_and_transforms(
                    name, pretrained="laion2b_s34b_b79k", device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(name)
                self.model.eval().to(device=self.device, dtype=self.dtype)
                self.out_dim = (
                    int(self.model.visual.output_dim)
                    if hasattr(self.model.visual, "output_dim")
                    else 512
                )
                self._has_text = True
                return
            except Exception:
                # fallback to resnet below
                pass

        # Fallback: torchvision resnet50 image-only
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.transform = T.Compose(
            [
                T.Resize(cfg.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(cfg.image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=ResNet50_Weights.IMAGENET1K_V2.meta["mean"],
                    std=ResNet50_Weights.IMAGENET1K_V2.meta["std"],
                ),
            ]
        )
        self.out_dim = 2048

    def preprocess(self, pil_image):
        return self.transform(pil_image)

    @torch.no_grad()
    def encode_images(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.to(device=self.device, dtype=self.dtype)
        if self._has_text:
            feats = self.model.encode_image(x)
            feats = feats.float()
        else:
            feats = self.model(x).float()
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if not self._has_text:
            raise RuntimeError(
                "Text queries require OpenCLIP. Install 'open-clip-torch' and use an open-clip backbone."
            )
        tokens = self.tokenizer(texts)
        x = tokens.to(device=self.device)
        feats = self.model.encode_text(x).float()
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats
