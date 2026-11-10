import io
import torch
import torchvision.utils as vutils
from PIL import Image
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from helper_lib.model import DiffusionModel
from helper_lib.generator import generate_diffusion_samples

router = APIRouter(prefix="/diffusion", tags=["Diffusion"])

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "trained_models/diffusion_cifar10.pth"

_model: DiffusionModel | None = None


def _load_model() -> DiffusionModel:
    global _model
    if _model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        m = DiffusionModel(
            img_channels  = checkpoint.get("img_channels",  3),
            base_channels = checkpoint.get("base_channels", 64),
            time_dim      = checkpoint.get("time_dim",      128),
            T             = checkpoint.get("T",             1000),
        )
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(DEVICE).eval()
        _model = m
    return _model


@router.get(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    summary="Generate CIFAR-10 images with the trained Diffusion model",
)
async def generate_images(
    num_samples: int = Query(default=9, ge=1, le=64),
    diffusion_steps: int = Query(default=100, ge=10, le=1000),
):
    try:
        model = _load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Diffusion weights not found at '{MODEL_PATH}'. Train first.",
        )

    samples = generate_diffusion_samples(model, DEVICE, num_samples, diffusion_steps)

    nrow = max(1, int(num_samples ** 0.5 + 0.5))
    grid = vutils.make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(grid_np, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")