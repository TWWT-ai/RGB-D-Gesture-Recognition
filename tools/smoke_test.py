"""CPU smoke test for the YOLOv2 anchor port. Run from src/: python ../smoke_test.py"""
import sys, os, math
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "src"), os.path.join(_here, "..", "src")):
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    raise SystemExit("cannot find src/ next to or above this script")

import torch as th
import numpy as np

from utils import (ANCHORS, anchor_tensor, shape_iou, encode_yolo_target,
                   decode_predictions, extract_best_pred_box, extract_gt_box,
                   compute_intersection_over_union, S)
from model import HandGestureNet, DetectionHead, load_handgesture_net
from train import yolo_detection_loss, compute_loss
import torch.nn as nn

th.manual_seed(0)
A = len(ANCHORS)
print(f"A = {A} anchors")

# ---- 1. forward shape + ranges -------------------------------------------
model = HandGestureNet(in_channels=4, n_classes=10, anchors=ANCHORS)
x = th.randn(2, 4, 448, 448)
det, seg, cls = model(x)
assert det.shape == (2, S, S, A * 5), det.shape
assert seg.shape == (2, 1, 448, 448), seg.shape
assert cls.shape == (2, 10), cls.shape
d = det.view(2, S, S, A, 5)
assert th.isfinite(det).all()
assert (d[..., 0] >= 0).all() and (d[..., 0] <= 1).all(), "cx out of [0,1]"
assert (d[..., 1] >= 0).all() and (d[..., 1] <= 1).all(), "cy out of [0,1]"
assert (d[..., 4] >= 0).all() and (d[..., 4] <= 1).all(), "conf out of [0,1]"
assert (d[..., 2] > 0).all() and (d[..., 3] > 0).all(), "w/h must be positive"
print("1. forward shapes + ranges           OK")

# ---- 2. centre really is locked inside its own cell ----------------------
# force huge raw t values -> sigmoid saturates -> centre must stay in the cell
head = DetectionHead(anchors=ANCHORS)
with th.no_grad():
    for m in head.net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.mul_(0)
            if m.bias is not None:
                m.bias.mul_(0).add_(50.0)   # sigmoid -> ~1
out = head(th.randn(1, 1024, S, S)).view(1, S, S, A, 5)
for r in range(S):
    for c in range(S):
        cx, cy = out[0, r, c, 0, 0].item(), out[0, r, c, 0, 1].item()
        assert c / S - 1e-4 <= cx <= (c + 1) / S + 1e-4, (r, c, cx)
        assert r / S - 1e-4 <= cy <= (r + 1) / S + 1e-4, (r, c, cy)
print("2. centre stays inside its own cell  OK")

# ---- 3. anchor assignment is by shape, and is stable ---------------------
anchors = anchor_tensor(ANCHORS)
# an anchor handed to itself must select itself, whatever ANCHORS currently is
idx = shape_iou(anchors, anchors).argmax(dim=1)
assert idx.tolist() == list(range(A)), idx.tolist()
# a box far bigger than every anchor must land on the largest one
biggest = int(th.argmax(anchors[:, 0] * anchors[:, 1]))
idx2 = shape_iou(th.tensor([[0.9, 0.9]]), anchors).argmax(dim=1)
assert idx2.item() == biggest, (idx2.item(), biggest)
print(f"3. shape-IoU assignment self-map + extremes  OK")

# ---- 4. loss is finite and gradients flow --------------------------------
target = th.zeros(2, S, S, 5)
target[0] = encode_yolo_target([0.30, 0.35, 0.55, 0.70])   # x1,y1,x2,y2
target[1] = encode_yolo_target([0.10, 0.10, 0.25, 0.30])
loss = yolo_detection_loss(det, target)
assert th.isfinite(loss), loss
loss.backward()
gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
assert gnorm > 0 and math.isfinite(gnorm)
print(f"4. det loss {loss.item():.4f}, grad norm {gnorm:.1f}   OK")

# ---- 5. multi-task compute_loss path -------------------------------------
model.zero_grad()
det, seg, cls = model(x)
targets = {"det": target, "Mask": th.zeros(2, 1, 448, 448),
           "Label": th.tensor([1, 2]), "has_mask": th.tensor([1.0, 0.0])}
total, parts = compute_loss((det, seg, cls), targets, nn.BCEWithLogitsLoss(),
                            nn.CrossEntropyLoss(), anchors=model.detecting_head.anchors)
assert all(math.isfinite(v) for v in parts.values()), parts
total.backward()
print(f"5. compute_loss {parts}   OK")

# ---- 6. all-unannotated batch must not crash -----------------------------
targets["has_mask"] = th.tensor([0.0, 0.0])
t2, p2 = compute_loss((det.detach(), seg.detach(), cls.detach()), targets,
                      nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(),
                      anchors=model.detecting_head.anchors)
assert p2["det"] == 0.0
print("6. batch with no annotated frame     OK")

# ---- 7. downstream helpers infer A from the tensor -----------------------
boxes = decode_predictions(det[0].detach(), conf_thresh=0.0)
assert len(boxes) == S * S * A, len(boxes)
best = extract_best_pred_box(det[0].detach())
assert best.shape == (4,)
gtb = extract_gt_box(target[0])
iou = compute_intersection_over_union(best, gtb)
assert 0.0 <= iou.item() <= 1.0
print(f"7. decode/extract (A inferred), IoU {iou.item():.3f}  OK")

# ---- 8. checkpoint round-trip keeps the anchors --------------------------
ck = {"model_state": model.state_dict(), "anchors": model.anchors_list}
th.save(ck, "/tmp/ck.pth")
ck2 = th.load("/tmp/ck.pth", map_location="cpu", weights_only=False)
m2 = load_handgesture_net(ck2, in_channels=4, n_classes=10)
assert m2.anchors_list == model.anchors_list
m2.eval(); model.eval()
with th.no_grad():
    assert th.allclose(m2(x)[0], model(x)[0], atol=1e-6)
print("8. checkpoint round-trip             OK")

# ---- 9. an old (anchor-less) checkpoint must fail LOUDLY -----------------
try:
    load_handgesture_net({"model_state": {}}, in_channels=4)
    raise AssertionError("should have raised")
except RuntimeError as e:
    assert "pre-YOLOv2" in str(e)
print("9. old checkpoint rejected loudly    OK")

# ---- 10. k-means machinery on synthetic data -----------------------------
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "src"), os.path.join(_here, "..", "src")):
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    raise SystemExit("cannot find src/ next to or above this script")
from kmeans_anchors import kmeans_iou, iou_wh, best_of_n_restarts, handpicked_baseline
rng = np.random.default_rng(0)
true = np.array([[0.12, 0.15], [0.30, 0.38], [0.60, 0.70]])
data = np.concatenate([true[i] * rng.lognormal(0, 0.08, size=(400, 2)) for i in range(3)])
cl, score = best_of_n_restarts(data, 3)
assert score > 0.85, score
found = np.sort(cl[:, 0])
assert np.allclose(found, np.sort(true[:, 0]), atol=0.03), (found, true[:, 0])
# more anchors must never be worse
prev = -1
for k in range(1, 7):
    _, s = best_of_n_restarts(data, k)
    assert s >= prev - 1e-6, (k, s, prev)
    prev = s
b = handpicked_baseline(data, 3)
assert b.shape == (3, 2) and np.allclose(b[:, 0], b[:, 1])
print(f"10. k-means recovers clusters, mean best-IoU {score:.3f}  OK")

print("\nAll smoke tests passed.")
