"""
routers/gan.py
--------------
FastAPI router that exposes a GAN image-generation endpoint.
Add to your existing FastAPI app with:

    from routers.gan import router as gan_router
    app.include_router(gan_router)
"""

import io
import torch
import torchvision.utils as vutils
from PIL import Image
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from helper_lib.model import GAN

router = APIRouter(prefix="/gan", tags=["GAN"])

# ---------------------------------------------------------------------------
# Load model once at import time
# ---------------------------------------------------------------------------
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "trained_models/gan_mnist.pth"

_model: GAN | None = None


def _load_model() -> GAN:
    global _model
    if _model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        latent_dim = checkpoint.get("latent_dim", 100)
        m = GAN(latent_dim=latent_dim)
        m.generator.load_state_dict(checkpoint["generator"])
        m.generator.to(DEVICE).eval()
        _model = m
    return _model


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    summary="Generate MNIST digits with the trained GAN",
)
async def generate_digits(
    num_samples: int = Query(default=9, ge=1, le=64, description="Number of digits to generate"),
):
    """Return a PNG image grid of GAN-generated MNIST digits."""
    try:
        model = _load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not found at '{MODEL_PATH}'. Run train_mnist_gan.py first.",
        )

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=DEVICE)
        fake_images = model.generator(z)   # (N, 1, 28, 28) in [-1, 1]

    nrow = max(1, int(num_samples ** 0.5 + 0.5))
    grid = vutils.make_grid(fake_images, nrow=nrow, normalize=True, value_range=(-1, 1))

    # Convert to uint8 grayscale PIL image
    grid_np = (grid[0].cpu().numpy() * 255).astype("uint8")   # single channel
    img = Image.fromarray(grid_np, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")
