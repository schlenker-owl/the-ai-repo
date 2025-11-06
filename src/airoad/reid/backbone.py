from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50


def _pick_device(user: Optional[str] = None) -> torch.device:
    """Choose device from user hint or auto-pick."""
    if user:
        if user == "cpu":
            return torch.device("cpu")
        if user == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if (user.isdigit() or user.startswith("cuda")) and torch.cuda.is_available():
            return torch.device("cuda:0" if user == "cuda" else f"cuda:{user}")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class BackboneConfig:
    name: str = "auto"  # "auto" | "open-clip:ViT-B-32" | "resnet50"
    device: Optional[str] = None
    dtype: str = "float32"  # "float32" | "float16" (fp16 on CPU is unreliable)
    image_size: int = 224  # typical for resnet/clip


class BaseBackbone(nn.Module):
    """Base interface for image-embedding backbones."""

    def __init__(self, device: torch.device, out_dim: int):
        super().__init__()
        self._device = device
        self.out_dim = out_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def preprocess(self, pil_or_ndarray):
        raise NotImplementedError

    @torch.no_grad()
    def encode_images(self, batch: torch.Tensor) -> torch.Tensor:
        """Return (B, D) L2-normalized features."""
        raise NotImplementedError


class TorchvisionResNet50(BaseBackbone):
    def __init__(self, device: torch.device, dtype: torch.dtype, image_size: int = 224):
        super().__init__(device, out_dim=2048)
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.model.eval().to(device=device, dtype=dtype)
        self.dtype = dtype
        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=ResNet50_Weights.IMAGENET1K_V2.meta["mean"],
                    std=ResNet50_Weights.IMAGENET1K_V2.meta["std"],
                ),
            ]
        )

    def preprocess(self, pil_or_ndarray):
        return self.transform(pil_or_ndarray)

    @torch.no_grad()
    def encode_images(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model(batch.to(device=self.device, dtype=self.dtype))
        return torch.nn.functional.normalize(feats, p=2, dim=-1)


class OpenCLIPBackbone(BaseBackbone):
    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype):
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError(
                "open-clip is not installed. Install with: uv add --group cv 'open-clip-torch>=2.24.0'"
            ) from e
        super().__init__(device, out_dim=512)  # most ViT-B produce 512-dim features
        name = model_name.split("open-clip:", 1)[1] if ":" in model_name else model_name
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            name, pretrained="laion2b_s34b_b79k", device=device
        )
        self.model.eval().to(device=device, dtype=dtype)
        self.dtype = dtype

    def preprocess(self, pil_or_ndarray):
        return self.transform(pil_or_ndarray)

    @torch.no_grad()
    def encode_images(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(batch.to(device=self.device, dtype=self.dtype))
        feats = feats.float()  # keep similarity math in fp32
        return torch.nn.functional.normalize(feats, p=2, dim=-1)


def build_backbone(cfg: BackboneConfig) -> BaseBackbone:
    """Factory that returns a ready-to-use backbone."""
    device = _pick_device(cfg.device)
    dtype = torch.float16 if (cfg.dtype == "float16" and device.type != "cpu") else torch.float32

    if cfg.name == "auto":
        # Prefer OpenCLIP if available; else ResNet50
        try:
            return OpenCLIPBackbone("open-clip:ViT-B-32", device, dtype)
        except Exception:
            return TorchvisionResNet50(device, dtype, image_size=cfg.image_size)

    if cfg.name.startswith("open-clip"):
        return OpenCLIPBackbone(cfg.name, device, dtype)
    if cfg.name == "resnet50":
        return TorchvisionResNet50(device, dtype, image_size=cfg.image_size)

    raise ValueError(f"Unknown backbone name: {cfg.name}")
