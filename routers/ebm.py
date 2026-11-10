import io
import torch
import torchvision.utils as vutils
from PIL import Image
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from helper_lib.model import EnergyModel
from helper_lib.generator import generate_ebm_samples

router = APIRouter(prefix="/ebm", tags=["EBM"])

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "trained_models/ebm_cifar10.pth"

_model: EnergyModel | None = None


def _load_model() -> EnergyModel:
    global _model
    if _model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        m = EnergyModel(img_channels=checkpoint.get("img_channels", 3))
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(DEVICE).eval()
        _model = m
    return _model


@router.get(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    summary="Generate CIFAR-10 images with the trained EBM",
)
async def generate_images(
    num_samples: int   = Query(default=9,     ge=1,  le=64),
    langevin_steps: int = Query(default=100,  ge=10, le=1000),
    step_size: float   = Query(default=10.0,  gt=0),
    noise_scale: float = Query(default=0.005, gt=0),
):
    try:
        model = _load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"EBM weights not found at '{MODEL_PATH}'. Train first.",
        )

    samples = generate_ebm_samples(model, DEVICE, num_samples,
                                   langevin_steps, step_size, noise_scale)

    nrow = max(1, int(num_samples ** 0.5 + 0.5))
    grid = vutils.make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(grid_np, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")