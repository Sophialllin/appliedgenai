from fastapi import FastAPI, HTTPException, UploadFile
import spacy
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from helper_lib.model import get_model
from helper_lib.data_loader import get_transform, CIFAR10_CLASSES

app = FastAPI()
nlp = spacy.load("en_core_web_lg")

_CHECKPOINT = Path(__file__).parent / "checkpoints" / "best_model.pth"
_cifar_model = get_model("CNN")
_ckpt = torch.load(_CHECKPOINT, map_location="cpu", weights_only=True)
_cifar_model.load_state_dict(_ckpt["model_state_dict"])
_cifar_model.eval()
_transform = get_transform()


@app.get("/")
def root():
    return {"message": "Hello, FastAPI with UV!"}


@app.get("/embedding")
def embedding(word: str):
    doc = nlp(word)
    if not doc.has_vector:
        raise HTTPException(status_code=404, detail="No vector for this word")
    return {
        "word": word,
        "dimension": len(doc.vector),
        "embedding": doc.vector.tolist(),
    }


@app.post("/predict_cifar")
async def predict_cifar(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")

    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = _cifar_model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    return {
        "class": CIFAR10_CLASSES[predicted.item()],
        "confidence": round(confidence.item(), 4),
    }
