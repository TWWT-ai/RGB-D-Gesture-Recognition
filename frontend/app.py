"""
FastAPI backend for the HandGestureNet demo.

Loads your trained 4-channel (RGB+Depth) model. The web app only sends RGB, so
we fake the depth channel with zeros. NOTE: accuracy is lower than with real
depth — this is the quick "just show the output" path.

Run:
    pip install -r requirements.txt
    python app.py
    # then open http://localhost:8000
"""
import io
import os
import sys
import base64

import numpy as np
import cv2
import torch as th
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# ---------------------------------------------------------------------------
# Paths  (edit these two lines only)
# ---------------------------------------------------------------------------
SRC_DIR = r"D:\School\Personal\UCL\COMP0248\CW1\Object Detection\src"
WEIGHTS = r"D:\School\Personal\UCL\COMP0248\CW1\data\weights\trained_outputs_final\best.pth"
IMAGE_SIZE = 448

# Make model.py / utils.py importable, then sanity-check both paths up front
sys.path.insert(0, SRC_DIR)
assert os.path.exists(os.path.join(SRC_DIR, "model.py")), f"model.py not found in: {SRC_DIR}"
assert os.path.exists(WEIGHTS), f"weights file not found: {WEIGHTS}"

from model import HandGestureNet, load_handgesture_net   # noqa: E402  # >>> [YOLOv2-CHANGED]
from utils import extract_best_pred_box   # noqa: E402

# Gesture names in label order (G01..G10)
CLASSES = ["call", "dislike", "like", "ok", "one",
           "palm", "peace", "rock", "stop", "three"]

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Depth model
depth_pipe = pipeline(task="depth-estimation",
                      model="depth-anything/Depth-Anything-V2-Small-hf",
                      device=0 if th.cuda.is_available() else -1)
print("[demo] Depth-Anything-V2-Small (HF) loaded")

# Main model
# Point WEIGHTS at a checkpoint produced on THIS branch: a checkpoint from
# another branch (main, yolov2..yolov5) has a different detection head and is
# rejected with a clear message. >>> [YOLOv10-CHANGED] the YOLOv10 head is
# NMS-free, so the most confident box is used directly.
checkpoint = th.load(WEIGHTS, map_location=device)
model_main = load_handgesture_net(checkpoint, in_channels=4, n_classes=10, device=device)
model_main.eval()
print(f"[demo] model loaded from {WEIGHTS} on {device}")


def preprocess(rgb_np):
    """RGB uint8 [H,W,3] -> (resized rgb for drawing, 4-channel tensor for model)."""
    rgb = cv2.resize(rgb_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb_norm = rgb.astype(np.float32) / 255.0

    # DA expects BGR; use the resized image so sizes line up
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_pil = depth_pipe(Image.fromarray(rgb))["depth"]
    depth = np.array(depth_pil).astype(np.float32) / 255.0        # -> [0,1]，越大=越近
    depth = 1.0 - depth                                           # 翻转：越大=越远（对齐训练约定）
    if depth.shape != (IMAGE_SIZE, IMAGE_SIZE):
        depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE))
    # depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    img = np.concatenate([rgb_norm, depth[..., None]], axis=-1)   # (448,448,4)
    tensor = th.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return rgb, tensor

@th.no_grad()
def run_inference(rgb_np):
    rgb_resized, x = preprocess(rgb_np)
    det, seg, cls = model_main(x.to(device))

    # --- classification ---
    probs = F.softmax(cls[0], dim=0)
    idx = int(probs.argmax())
    label, conf = CLASSES[idx], float(probs[idx])

    # --- detection: best box (cx,cy,w,h normalised) -> pixel corners ---
    cx, cy, w, h = [float(v) for v in extract_best_pred_box(det[0].cpu())]
    x1, y1 = int((cx - w / 2) * IMAGE_SIZE), int((cy - h / 2) * IMAGE_SIZE)
    x2, y2 = int((cx + w / 2) * IMAGE_SIZE), int((cy + h / 2) * IMAGE_SIZE)

    # --- segmentation: logits -> binary mask ---
    mask = th.sigmoid(seg[0, 0]).cpu().numpy() > 0.5

    # --- draw everything on the resized RGB ---
    out = rgb_resized.copy()
    overlay = out.copy()
    overlay[mask] = (255, 80, 80)                         # red hand overlay
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"{label} {conf:.2f}", (max(0, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return out, label, conf, [x1, y1, x2, y2]


app = FastAPI(title="HandGesture Demo")
# Allow a separate React dev server (e.g. Vite on :5173) to call the API
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    out, label, conf, box = run_inference(rgb)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode()
    return JSONResponse({
        "label": label,
        "confidence": round(conf, 4),
        "box": box,
        "image": f"data:image/png;base64,{b64}",
    })


@app.get("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)