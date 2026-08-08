# PulmoScan AI — Setup, Deployment & Run Guide

Stacking Ensemble (DenseNet121 + ResNet50 + EfficientNetB0 + MobileNetV2 → Logistic Regression)
for TB & Pneumonia detection from chest X-rays.

---

## Project Structure

```
tbpneumonia/
├── main.py                   ← FastAPI app (inference pipeline + routes)
├── requirements.txt
├── models/                   ← PUT YOUR MODEL FILES HERE
│   ├── densenet121.pth
│   ├── resnet50.pth
│   ├── efficientnetb0.pth
│   ├── mobilenetv2.pth
│   └── ensemble_meta.pkl
├── static/                  ← Vercel frontend output
│   ├── index.html
│   ├── config.js            ← Render API URL configuration
│   ├── css/style.css
│   └── js/app.js
└── templates/
    └── index.html
```

---

## Step 1 — Create a Virtual Environment

```bash
# Inside the tbpneumonia/ folder:
python -m venv venv

# Activate it:
# macOS / Linux:
source venv/bin/activate

# Windows (CMD):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** If you have CUDA, install the matching PyTorch build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Then run `pip install -r requirements.txt` (torch/torchvision won't be reinstalled).

---

## Step 3 — Place Your Model Files

Create a `models/` folder inside `tbpneumonia/` and copy your files in:

```bash
mkdir models
```

Then copy your files so the structure looks like:

```
models/
├── densenet121.pth
├── resnet50.pth
├── efficientnetb0.pth
├── mobilenetv2.pth
└── ensemble_meta.pkl
```

### If your .pth files have different names

Open `main.py` and update the path constants near the top:

```python
DENSENET_PATH     = MODEL_DIR / "your_densenet_filename.pth"
RESNET_PATH       = MODEL_DIR / "your_resnet_filename.pth"
EFFICIENTNET_PATH = MODEL_DIR / "your_efficientnet_filename.pth"
MOBILENET_PATH    = MODEL_DIR / "your_mobilenet_filename.pth"
ENSEMBLE_PATH     = MODEL_DIR / "your_ensemble_filename.pkl"
```

### If your models are stored elsewhere

Set the `MODEL_DIR` environment variable before running:

```bash
# macOS/Linux:
export MODEL_DIR=/path/to/your/models

# Windows CMD:
set MODEL_DIR=C:\path\to\your\models
```

---

## Step 4 — Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser at: **http://localhost:8000**

The `--reload` flag restarts the server automatically when you edit code.
Remove it for a cleaner presentation run.

## Deploy to Render + Vercel

The backend and frontend deploy independently:

1. Push this repository to GitHub. The model weights in `models/` are required by Render and are intentionally tracked; do not upload the included project ZIP or virtual environment.
2. In Render, create a **Blueprint** from the repository. It detects `render.yaml`; choose a unique service name and set `FRONTEND_ORIGINS` to the Vercel URL after the frontend exists.
3. In Vercel, import the same repository and deploy with the repository root as the **Root Directory**. It serves `static/` as a static site using `vercel.json`.
4. Edit `static/config.js` before the Vercel deployment and set the Render API URL:

```js
window.PULMOSCAN_API_URL = 'https://your-render-service.onrender.com';
```

5. Redeploy Vercel after changing `config.js`. Update Render's `FRONTEND_ORIGINS` with the final Vercel URL, then redeploy Render.

For a local all-in-one run, leave `PULMOSCAN_API_URL` blank; the UI calls the local FastAPI server.

---

## Step 5 — Verify Models Loaded

Visit **http://localhost:8000/api/status** — you should see:

```json
{
  "models_loaded": true,
  "base_models": ["densenet", "resnet", "efficientnet", "mobilenet"],
  "ensemble_ready": true,
  "device": "cpu",
  "load_errors": {}
}
```

If any model failed to load, the reason is shown in `load_errors`.

---

## Troubleshooting

### "File not found" for a .pth file
- Double-check the filename matches exactly (case-sensitive on Linux/Mac).
- Make sure the file is inside the `models/` folder.

### State dict key mismatch error
Your checkpoint may have been saved with `torch.save({"model_state_dict": model.state_dict(), ...})`.
The loader in `main.py` handles the common wrappers `model_state_dict` and `state_dict` automatically.
If your key is different, update the `load_model()` function in `main.py`.

### Grad-CAM produces a blank/black heatmap
This can happen if the target layer name changed. Check `get_target_layer()` in `main.py`
and verify it matches your exact model architecture.

### Port already in use
```bash
uvicorn main:app --port 8001
```

### CUDA out of memory
Add `--workers 1` to the uvicorn command, or set `DEVICE = torch.device("cpu")` in `main.py`.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/status` | GET | Model load status + device info |
| `/api/predict` | POST | Upload image, returns full prediction JSON |

### `/api/predict` Response Format

```json
{
  "predicted_class": "TUBERCULOSIS",
  "ensemble_probs": {
    "NORMAL": 0.012,
    "PNEUMONIA": 0.043,
    "TUBERCULOSIS": 0.921,
    "UNKNOWN": 0.024
  },
  "ensemble_conf": 0.921,
  "base_predictions": {
    "DenseNet121":    { "class": "TUBERCULOSIS", "confidence": 0.94, "probs": {...} },
    "ResNet50":       { "class": "TUBERCULOSIS", "confidence": 0.88, "probs": {...} },
    "EfficientNetB0": { "class": "TUBERCULOSIS", "confidence": 0.91, "probs": {...} },
    "MobileNetV2":    { "class": "TUBERCULOSIS", "confidence": 0.87, "probs": {...} }
  },
  "gradcam_image":  "<base64 PNG>",
  "saliency_image": "<base64 PNG>",
  "gradcam_model":  "DenseNet121",
  "device":         "cpu",
  "load_errors":    {}
}
```
