"""
TB & Pneumonia Detection System — FastAPI Backend
Stacking Ensemble: DenseNet121 + ResNet50 + EfficientNetB0 + MobileNetV2 → Logistic Regression
"""

import io
import os
import uuid
import pickle
import base64
import traceback
import gc
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
TARGET_CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
CLASS_COLORS = {
    "NORMAL":       "#22c55e",   # green
    "PNEUMONIA":    "#f97316",   # orange
    "TUBERCULOSIS": "#ef4444",   # red
    "UNKNOWN":      "#a855f7",   # purple
}

# Paths are anchored to this source file so they work locally and on Render.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models")).expanduser().resolve()
DENSENET_PATH     = MODEL_DIR / "model1_densenet_best.pth"
RESNET_PATH       = MODEL_DIR / "model2_resnet50_4class.pth"
EFFICIENTNET_PATH = MODEL_DIR / "model3_efficientnet_4class.pth"
MOBILENET_PATH    = MODEL_DIR / "model4_mobilenet_4class.pth"
ENSEMBLE_PATH     = MODEL_DIR / "ensemble_meta_model.pkl"

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet normalization (matches training preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────
#  MODEL BUILDERS  (mirrors training code)
# ─────────────────────────────────────────────
def build_densenet(num_classes=NUM_CLASSES):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def build_resnet(num_classes=NUM_CLASSES):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_efficientnet(num_classes=NUM_CLASSES):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def build_mobilenet(num_classes=NUM_CLASSES):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients  = self.gradients[0]            # [C, H, W]
        activations = self.activations[0]         # [C, H, W]
        weights = gradients.mean(dim=(1, 2))      # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def get_target_layer(model_name: str, model):
    """Return the final convolutional layer for each architecture."""
    if model_name == "densenet":
        return model.features.denseblock4.denselayer16.conv2
    elif model_name == "resnet":
        return model.layer4[-1].conv3
    elif model_name == "efficientnet":
        return model.features[-1][0]
    elif model_name == "mobilenet":
        return model.features[-1][0]
    raise ValueError(f"Unknown model: {model_name}")


def generate_gradcam_image(model, model_name: str, input_tensor, class_idx: int, orig_img_array: np.ndarray) -> str:
    """Returns base64-encoded PNG of GradCAM heatmap overlay."""
    model.eval()
    layer = get_target_layer(model_name, model)
    gcam  = GradCAM(model, layer)

    t = input_tensor.clone().requires_grad_(True)
    cam = gcam.generate(t, class_idx)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_resized = cv2.resize(orig_img_array, (224, 224))
    if orig_resized.ndim == 2:
        orig_resized = np.stack([orig_resized] * 3, axis=-1)

    overlay = (0.5 * orig_resized.astype(np.float32) + 0.5 * heatmap.astype(np.float32)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0f1117")
    titles = ["Original X-Ray", "Grad-CAM Heatmap", "Overlay"]
    imgs   = [orig_resized, heatmap, overlay]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
        ax.axis("off")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_saliency_image(model, input_tensor, class_idx: int, orig_img_array: np.ndarray) -> str:
    """Pixel saliency via input-gradient magnitude."""
    model.eval()
    t = input_tensor.clone().requires_grad_(True)
    output = model(t)
    model.zero_grad()
    output[0, class_idx].backward()

    saliency = t.grad.data.abs().squeeze()           # [3, 224, 224]
    saliency, _ = torch.max(saliency, dim=0)         # [224, 224]
    saliency = saliency.numpy()
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    orig_resized = cv2.resize(orig_img_array, (224, 224))
    if orig_resized.ndim == 2:
        orig_resized = np.stack([orig_resized] * 3, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor("#0f1117")
    axes[0].imshow(orig_resized)
    axes[0].set_title("Original X-Ray", color="white", fontsize=9, fontweight="bold", pad=6)
    axes[0].axis("off")
    axes[1].imshow(orig_resized)
    axes[1].imshow(saliency, cmap="hot", alpha=0.55)
    axes[1].set_title("Pixel Saliency", color="white", fontsize=9, fontweight="bold", pad=6)
    axes[1].axis("off")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────
#  APP STARTUP  — load models once
# ─────────────────────────────────────────────
app = FastAPI(title="TB & Pneumonia Detection System")

# Set FRONTEND_ORIGINS on Render to your Vercel URL, for example:
# https://your-project.vercel.app.  Comma-separated values are supported.
frontend_origins = [
    origin.strip() for origin in os.environ.get("FRONTEND_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

models_loaded = False
base_models   = {}
meta_learner  = None
load_errors   = {}

MODEL_SPECS = [
    ("densenet", build_densenet, DENSENET_PATH),
    ("resnet", build_resnet, RESNET_PATH),
    ("efficientnet", build_efficientnet, EFFICIENTNET_PATH),
    ("mobilenet", build_mobilenet, MOBILENET_PATH),
]


def load_model(builder_fn, path, name):
    try:
        m = builder_fn()
        state = torch.load(path, map_location=DEVICE)
        # Handle common checkpoint wrappers
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
        m.to(DEVICE)
        m.eval()
        return m, None
    except Exception as e:
        return None, str(e)


@app.on_event("startup")
async def startup_event():
    global base_models, meta_learner, models_loaded, load_errors

    # Model weights are intentionally loaded per request rather than at startup.
    # This keeps the four-model ensemble within Render's 512 MB free-instance
    # memory limit while preserving identical inference behaviour.
    base_models = {}
    for name, _, path in MODEL_SPECS:
        if not path.exists():
            load_errors[name] = f"File not found: {path}"
            print(f"  ✗ {name}: {path} not found")

    if ENSEMBLE_PATH.exists():
            try:
                with open(ENSEMBLE_PATH, "rb") as f:
                    meta_learner = pickle.load(f)
            except Exception:
                import joblib
                meta_learner = joblib.load(ENSEMBLE_PATH)
    else:
        load_errors["ensemble"] = f"File not found: {ENSEMBLE_PATH}"
        print(f"  ✗ Ensemble: {ENSEMBLE_PATH} not found")

    models_loaded = not any(name in load_errors for name, _, _ in MODEL_SPECS)


# ─────────────────────────────────────────────
#  INFERENCE CORE
# ─────────────────────────────────────────────
def run_inference(pil_image: Image.Image) -> dict:
    """Full pipeline: image → base learner probs → meta-learner → result."""
    orig_array = np.array(pil_image.convert("RGB"))
    tensor     = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    model_order = ["densenet", "resnet", "efficientnet", "mobilenet"]
    model_labels = {
        "densenet":     "DenseNet121",
        "resnet":       "ResNet50",
        "efficientnet": "EfficientNetB0",
        "mobilenet":    "MobileNetV2",
    }

    # --- Base learner inference ---
    base_probs  = {}
    base_preds  = {}
    prob_vectors = []

    spec_by_name = {name: (builder, path) for name, builder, path in MODEL_SPECS}

    for name in model_order:
        builder, path = spec_by_name[name]
        model, err = load_model(builder, path, name)
        if model is None:
            load_errors[name] = err or "Unable to load model"
            prob_vectors.append([0.25, 0.25, 0.25, 0.25])   # fallback uniform
            base_probs[name] = [0.25] * 4
            base_preds[name] = {"class": "N/A", "confidence": 0.0}
            continue

        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        del model
        gc.collect()

        prob_vectors.append(probs.tolist())
        base_probs[name] = probs.tolist()
        pred_idx = int(np.argmax(probs))
        base_preds[name] = {
            "class":      TARGET_CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probs":      {TARGET_CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }

    # --- Ensemble meta-learner ---
    meta_input = np.array(prob_vectors).flatten().reshape(1, -1)  # [1, 16]

    if meta_learner is not None:
        ensemble_probs = meta_learner.predict_proba(meta_input)[0]
        ensemble_pred  = int(np.argmax(ensemble_probs))
    else:
        # Fallback: average
        ensemble_probs = np.mean(prob_vectors, axis=0)
        ensemble_pred  = int(np.argmax(ensemble_probs))

    predicted_class = TARGET_CLASSES[ensemble_pred]

    # --- Grad-CAM on the best individual model ---
    gradcam_b64  = None
    saliency_b64 = None
    gradcam_model = None
    gradcam_name  = None

    # Pick whichever base model agrees with ensemble (highest conf for predicted class)
    best_score = -1
    for name in model_order:
        if name in base_probs:
            score = base_probs[name][ensemble_pred]
            if score > best_score:
                best_score    = score
                gradcam_name  = name

    if gradcam_name is not None:
        try:
            builder, path = spec_by_name[gradcam_name]
            gradcam_model, err = load_model(builder, path, gradcam_name)
            if gradcam_model is None:
                raise RuntimeError(err or "Unable to load model for explainability")
            t_grad = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
            gradcam_b64  = generate_gradcam_image(gradcam_model, gradcam_name, t_grad, ensemble_pred, orig_array)
            t_grad2 = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
            saliency_b64 = generate_saliency_image(gradcam_model, t_grad2, ensemble_pred, orig_array)
        except Exception as e:
            print(f"XAI generation error: {e}")
            traceback.print_exc()
        finally:
            if gradcam_model is not None:
                del gradcam_model
            gc.collect()

    return {
        "predicted_class":  predicted_class,
        "ensemble_probs":   {TARGET_CLASSES[i]: float(ensemble_probs[i]) for i in range(NUM_CLASSES)},
        "ensemble_conf":    float(ensemble_probs[ensemble_pred]),
        "base_predictions": {model_labels[n]: base_preds[n] for n in model_order},
        "gradcam_image":    gradcam_b64,
        "saliency_image":   saliency_b64,
        "gradcam_model":    model_labels.get(gradcam_name, ""),
        "device":           str(DEVICE),
        "load_errors":      load_errors,
    }


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":      request,
        "models_ready": models_loaded,
        "load_errors":  load_errors,
    })


@app.get("/api/status")
async def status():
    return {
        "models_loaded":   models_loaded,
        "base_models":     [name for name, _, _ in MODEL_SPECS if name not in load_errors],
        "ensemble_ready":  meta_learner is not None,
        "device":          str(DEVICE),
        "load_errors":     load_errors,
    }


@app.get("/health")
async def health():
    """Lightweight endpoint used by Render health checks."""
    return {"status": "ok", "models_loaded": models_loaded}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Check /api/status for details.")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 20 MB.")

    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid X-ray image.")

    try:
        result = run_inference(pil_image)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return JSONResponse(content=result)
