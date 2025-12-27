from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms as T

import matplotlib.pyplot as plt
import cv2

try:
    import timm
except Exception:
    timm = None


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_image_rgb(path: str | Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))


def default_transform(img_size: int):
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def overlay_heatmap_on_image(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    hm = cv2.applyColorMap(heatmap_uint8, colormap)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(hm, alpha, img, 1 - alpha, 0)
    return out


def resize_cam_to_image(cam: np.ndarray, img_size: int) -> np.ndarray:
    cam = cv2.resize(cam, (img_size, img_size))
    cam = np.maximum(cam, 0)
    m = cam.max()
    if m > 0:
        cam = cam / (m + 1e-12)
    return cam


def save_overlay_png(rgb_np: np.ndarray, heatmap: np.ndarray, out_path: str | Path, alpha: float = 0.45):
    overlay = overlay_heatmap_on_image(rgb_np, heatmap, alpha=alpha)
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


# ----------------------------
# Model loading (timm ViT)
# ----------------------------
def load_vit_model(
    model_name: str = "vit_base_patch16_224",
    checkpoint_path: Optional[str | Path] = None,
    device_str: str = "cpu",
) -> nn.Module:
    if timm is None:
        raise RuntimeError("timm is not installed. Add timm to requirements.txt")

    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")

    # num_classes=2: OK vs NOK (binary demo)
    model = timm.create_model(model_name, pretrained=True, num_classes=2)

    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        # support {state_dict: ...} or raw state dict
        state = ckpt.get("state_dict", ckpt)
        # common prefix cleanup
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)

    model.eval()
    model.to(device)
    return model


# ----------------------------
# Grad-CAM on last ViT block
# ----------------------------
def gradcam_last_block(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    acts = []
    grads = []

    def fwd_hook(_m, _i, o):
        # o expected: [B, N, C] for ViT blocks
        acts.append(o)

    def bwd_hook(_m, _gi, go):
        # go is tuple of gradients w.r.t outputs
        grads.append(go[0])

    # timm ViT typically has model.blocks
    block = model.blocks[-1]

    h1 = block.register_forward_hook(fwd_hook)
    # safer than deprecated register_backward_hook
    h2 = block.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    loss = logits[0, target_class]
    loss.backward()

    h1.remove()
    h2.remove()

    a = acts[0].detach()    # [B, N, C]
    g = grads[0].detach()   # [B, N, C]

    # average gradients over tokens
    weights = g.mean(dim=1, keepdim=True)      # [B, 1, C]
    cam_tokens = (a * weights).sum(dim=-1)     # [B, N]
    cam_tokens = cam_tokens[0]

    # drop CLS token if present
    if cam_tokens.numel() > 1:
        cam_tokens = cam_tokens[1:]

    cam = cam_tokens.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    m = cam.max()
    if m > 0:
        cam = cam / (m + 1e-12)

    # reshape token map into square
    n = cam.shape[0]
    hw = int(np.sqrt(n))
    if hw * hw != n:
        # fallback
        return np.zeros((224, 224), dtype=np.float32)

    cam = cam.reshape(hw, hw)
    return cam.astype(np.float32)


# ----------------------------
# Attention map (optional)
# ----------------------------
def vit_last_attn_map(model: nn.Module, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """
    Many timm ViT models do not expose attention weights by default.
    This function tries to read them if available; otherwise returns None.
    """
    with torch.no_grad():
        _ = model(input_tensor)

    blk = getattr(model, "blocks", None)
    if not blk:
        return None

    last = blk[-1]
    attn = getattr(getattr(last, "attn", None), "weights", None)
    if isinstance(attn, torch.Tensor):
        return attn.detach().cpu().numpy()

    # Some custom models store last attn somewhere else
    last_attn = getattr(model, "_last_attn", None)
    if isinstance(last_attn, torch.Tensor):
        return last_attn.detach().cpu().numpy()

    return None


def generate_gradcam(
    model: nn.Module,
    img_path: str | Path,
    img_size: int = 224,
    device_str: str = "cpu",
    target_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")

    img = load_image_rgb(img_path)
    tfm = default_transform(img_size)
    tens = tfm(img).unsqueeze(0).to(device)

    cam_tokens = gradcam_last_block(model, tens, target_class=target_class)
    cam = resize_cam_to_image(cam_tokens, img_size)

    rgb_np = np.array(img.resize((img_size, img_size)))
    return cam, rgb_np


def generate_attnmap(
    model: nn.Module,
    img_path: str | Path,
    img_size: int = 224,
    device_str: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")

    img = load_image_rgb(img_path)
    tfm = default_transform(img_size)
    tens = tfm(img).unsqueeze(0).to(device)

    attn = vit_last_attn_map(model, tens)
    if attn is None:
        # fallback: zeros
        a = np.zeros((img_size, img_size), dtype=np.float32)
        rgb_np = np.array(img.resize((img_size, img_size)))
        return a, rgb_np

    # attn shape often: [B, heads, tokens, tokens]
    attn_mean = attn.mean(axis=1)  # [B, tokens, tokens]
    vec = attn_mean[0, 0, 1:]      # CLS -> patches
    hw = int(np.sqrt(vec.size))
    if hw * hw != vec.size:
        a = np.zeros((img_size, img_size), dtype=np.float32)
    else:
        a = vec.reshape(hw, hw).astype(np.float32)
        a = resize_cam_to_image(a, img_size)

    rgb_np = np.array(img.resize((img_size, img_size)))
    return a, rgb_np


def save_combined_gradcam(
    model: nn.Module,
    img_path: str | Path,
    out_path: str | Path,
    img_size: int = 224,
    device_str: str = "cpu",
    alpha: float = 0.45,
):
    cam, rgb_np = generate_gradcam(model, img_path, img_size=img_size, device_str=device_str, target_class=1)
    save_overlay_png(rgb_np, cam, out_path, alpha=alpha)


def save_combined_attnmap(
    model: nn.Module,
    img_path: str | Path,
    out_path: str | Path,
    img_size: int = 224,
    device_str: str = "cpu",
    alpha: float = 0.45,
):
    attn, rgb_np = generate_attnmap(model, img_path, img_size=img_size, device_str=device_str)
    save_overlay_png(rgb_np, attn, out_path, alpha=alpha)


def process_images(
    model: nn.Module,
    ok_paths: List[str | Path],
    nok_path: str | Path,
    out_dir: str | Path,
    mode: str = "gradcam",
    img_size: int = 224,
    device_str: str = "cpu",
    alpha: float = 0.45,
) -> List[Path]:
    """
    Creates heatmaps for each OK image and for the NOK image.
    Returns list of output PNG paths.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    outputs: List[Path] = []

    def run_one(p: str | Path):
        p = Path(p)
        name = p.stem
        if mode == "gradcam":
            op = out_dir / f"{name}_gradcam.png"
            save_combined_gradcam(model, p, op, img_size=img_size, device_str=device_str, alpha=alpha)
        else:
            op = out_dir / f"{name}_attn.png"
            save_combined_attnmap(model, p, op, img_size=img_size, device_str=device_str, alpha=alpha)
        outputs.append(op)

    for p in ok_paths:
        run_one(p)

    run_one(nok_path)
    return outputs
