import torch
import timm

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def load_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 2,
):
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )

    model.eval()
    model.to(_DEVICE)

    _MODEL = model
    return _MODEL


def get_device():
    return _DEVICE
