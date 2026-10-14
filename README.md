# SPS GenAI API

FastAPI service with a spaCy word-embedding endpoint and a CIFAR-10 image classifier.

## Requirements

- Python 3.14 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
uv sync
```

## Train the CIFAR-10 classifier

The checkpoint in `checkpoints/best_model.pth` was trained on Google Colab (GPU) using the
same `train.py` script. Results: **best validation accuracy 66.12%** at epoch 6 of 10,
**test accuracy 66.83%**.

To retrain locally (downloads CIFAR-10 ~170 MB to `data/`):

```bash
uv run python train.py --epochs 10
```

Optional flags: `--batch-size 64`, `--lr 0.001`, `--data-dir data`, `--checkpoint-dir checkpoints`.

## Start the API

Development server (auto-reload):

```bash
uv run fastapi dev main.py
```

Production server:

```bash
uv run fastapi run main.py --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/embedding?word=<word>` | 300-d spaCy word vector |
| POST | `/predict_cifar` | CIFAR-10 class prediction for an uploaded image |

## Example: predict a CIFAR-10 image

```bash
curl -X POST http://127.0.0.1:8000/predict_cifar \
  -F "file=@path/to/image.jpg"
```

Response:

```json
{
  "class": "dog",
  "confidence": 0.8731
}
```

The model accepts any image format supported by Pillow (JPEG, PNG, BMP, …).
It resizes the image to 64×64 and normalises with mean/std 0.5 before inference.
Returns HTTP 400 for non-image or corrupt files.
