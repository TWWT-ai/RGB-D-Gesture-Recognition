"""CPU smoke test for the YOLOv4 port. Run from src/: python ../tools/smoke_test.py"""
import sys, os, math, tempfile
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "src"), os.path.join(_here, "..", "src")):
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    raise SystemExit("cannot find src/ next to or above this script")

import torch as th
import numpy as np

from utils import (ANCHORS, GRIDS, anchor_tensor, shape_iou, encode_yolo_target,
                   decode_predictions, extract_best_pred_box, extract_gt_box,
                   compute_intersection_over_union, split_anchors_by_scale, S,
                   bbox_ciou)
from model import HandGestureNet, DetectionHead, load_handgesture_net, ARCH, SPP
from train import yolo_detection_loss, compute_loss, gt_boxes_from_target, assign_targets
import torch.nn as nn

th.manual_seed(0)
A = len(ANCHORS)
groups = split_anchors_by_scale(ANCHORS)
M = sum(g * g * len(grp) for g, grp in zip(GRIDS, groups))
print(f"A = {A} anchors over grids {GRIDS} as {[len(g) for g in groups]}, M = {M} boxes")

# ---- 0. anchor split: biggest on the coarsest grid, nothing lost ----------
assert sum(len(g) for g in groups) == A
areas = [[w * h for w, h in g] for g in groups]
for coarse, fine in zip(areas, areas[1:]):
    assert min(coarse) >= max(fine), areas
assert [len(g) for g in split_anchors_by_scale([(0.1 * i, 0.1 * i) for i in range(1, 10)])] == [3, 3, 3]
print("0. anchor -> grid split               OK")

# ---- 1. forward shape + ranges -------------------------------------------
model = HandGestureNet(in_channels=4, n_classes=10, anchors=ANCHORS)
x = th.randn(2, 4, 448, 448)
det, seg, cls = model(x)
assert det.shape == (2, M, 5), det.shape
assert seg.shape == (2, 1, 448, 448), seg.shape
assert cls.shape == (2, 10), cls.shape
d = det
assert th.isfinite(det).all()
# YOLOv4 grid-sensitivity decode may overshoot the image by (scale_xy-1)/2 cells
slack = max((DetectionHead.SCALE_XY[g] - 1) / 2 / g for g in GRIDS) + 1e-6
assert (d[..., 0] >= -slack).all() and (d[..., 0] <= 1 + slack).all(), "cx out of range"
assert (d[..., 1] >= -slack).all() and (d[..., 1] <= 1 + slack).all(), "cy out of range"
assert (d[..., 4] >= 0).all() and (d[..., 4] <= 1).all(), "conf out of [0,1]"
assert (d[..., 2] > 0).all() and (d[..., 3] > 0).all(), "w/h must be positive"
# objectness prior: everything starts near p = 0.01, not 0.5
assert d[..., 4].mean() < 0.1, d[..., 4].mean()
print("1. forward shapes + ranges           OK")

# ---- 2. centre really is locked inside its own cell, on every grid -------
head = DetectionHead(anchors=ANCHORS, in_channels=model.neck.out_channels)
with th.no_grad():
    for m in head.preds.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.mul_(0)
            if m.bias is not None:
                m.bias.mul_(0).add_(50.0)   # sigmoid -> ~1
feats = [th.randn(1, c, g, g) for c, g in zip(model.neck.out_channels, GRIDS)]
out = head(feats)
lay = head.layout()
for a in range(A):
    Sa = int(lay["grid"][a]); off = int(lay["offset"][a])
    tol = (DetectionHead.SCALE_XY[Sa] - 1) / 2 / Sa + 1e-4
    cnt = int(lay["count"][a]); loc = int(lay["local"][a])
    for r in range(Sa):
        for c in range(Sa):
            cx, cy = out[0, off + (r * Sa + c) * cnt + loc, :2].tolist()
            assert c / Sa - tol <= cx <= (c + 1) / Sa + tol, (a, r, c, cx)
            assert r / Sa - tol <= cy <= (r + 1) / Sa + tol, (a, r, c, cy)
            # and with sigmoid -> 1 the stretched decode DOES pass the border
            assert cx > (c + 1) / Sa, (a, r, c, cx)
print("2. centre locked to its cell (+ grid-sensitivity slack)  OK")

# ---- 3. anchor assignment is by shape, and is stable ---------------------
anchors = anchor_tensor(ANCHORS)
# an anchor handed to itself must select itself, whatever ANCHORS currently is
idx = shape_iou(anchors, anchors).argmax(dim=1)
assert idx.tolist() == list(range(A)), idx.tolist()
# a box far bigger than every anchor must land on the largest one -> 7x7 grid
idx2 = shape_iou(th.tensor([[0.9, 0.9]]), lay["anchors"]).argmax(dim=1)
assert int(lay["grid"][idx2]) == GRIDS[0], int(lay["grid"][idx2])
idx3 = shape_iou(th.tensor([[0.01, 0.01]]), lay["anchors"]).argmax(dim=1)
assert int(lay["grid"][idx3]) == GRIDS[-1], int(lay["grid"][idx3])
print(f"3. shape-IoU assignment picks the grid too  OK")

# ---- 4. loss is finite and gradients flow --------------------------------
target = th.zeros(2, S, S, 5)
target[0] = encode_yolo_target([0.30, 0.35, 0.55, 0.70])   # x1,y1,x2,y2
target[1] = encode_yolo_target([0.10, 0.10, 0.25, 0.30])
g = gt_boxes_from_target(target)
assert th.allclose(g[0, :4], th.tensor([0.425, 0.525, 0.25, 0.35])), g
loss = yolo_detection_loss(det, target, model.detecting_head.layout())
assert th.isfinite(loss), loss
loss.backward()
gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
assert gnorm > 0 and math.isfinite(gnorm)
print(f"4. det loss {loss.item():.4f}, grad norm {gnorm:.1f}   OK")

# ---- 4b. a perfect prediction gives (almost) zero box loss ---------------
lay = model.detecting_head.layout()
fake = th.zeros(1, M, 5); fake[..., 2:4] = 0.1
gt = th.tensor([0.43, 0.52, 0.2, 0.3])
box_idx, anc, fi = assign_targets(gt[None], lay)
assert len(fi) >= 1 and len(set(fi.tolist())) == len(fi)
fake[0, fi, :4] = gt; fake[0, fi, 4] = 1.0
tgt = th.zeros(1, S, S, 5); tgt[0, 3, 3] = th.tensor([*gt.tolist(), 1.0])
l_perfect = yolo_detection_loss(fake, tgt, lay).item()
fake[0, fi[0], 0] += 0.02
l_off = yolo_detection_loss(fake, tgt, lay).item()
assert l_perfect < 1e-3 and l_off > l_perfect, (l_perfect, l_off)
print(f"4b. {len(fi)} positives; perfect -> loss {l_perfect:.2e}, nudged -> {l_off:.2e}   OK")

# ---- 4c. CIoU ------------------------------------------------------------
b1 = th.tensor([0.5, 0.5, 0.2, 0.3])
assert abs(bbox_ciou(b1, b1).item() - 1.0) < 1e-5
far = th.tensor([0.9, 0.9, 0.05, 0.05], requires_grad=True)
c = bbox_ciou(far, b1)
assert c.item() < 0 and compute_intersection_over_union(far, b1).item() == 0
c.backward()
assert far.grad.abs().sum() > 0, "CIoU must still give a gradient with zero overlap"
assert abs(bbox_ciou(b1 * 0.5, th.tensor([0.26, 0.25, 0.1, 0.15])).item() -
           bbox_ciou(b1, th.tensor([0.52, 0.5, 0.2, 0.3])).item()) < 1e-4, "scale-invariant"
print("4c. CIoU: 1 for identical, <0 and still a gradient when disjoint, scale-invariant  OK")

# ---- 4d. SPP keeps the spatial size and quadruples the channels ----------
y = SPP()(th.randn(1, 8, 7, 7))
assert y.shape == (1, 32, 7, 7), y.shape
print("4d. SPP shape                         OK")

# ---- 5. multi-task compute_loss path -------------------------------------
model.zero_grad()
det, seg, cls = model(x)
targets = {"det": target, "Mask": th.zeros(2, 1, 448, 448),
           "Label": th.tensor([1, 2]), "has_mask": th.tensor([1.0, 0.0])}
total, parts = compute_loss((det, seg, cls), targets, nn.BCEWithLogitsLoss(),
                            nn.CrossEntropyLoss(), layout=model.detecting_head.layout())
assert all(math.isfinite(v) for v in parts.values()), parts
total.backward()
print(f"5. compute_loss {parts}   OK")

# ---- 6. all-unannotated batch must not crash -----------------------------
targets["has_mask"] = th.tensor([0.0, 0.0])
t2, p2 = compute_loss((det.detach(), seg.detach(), cls.detach()), targets,
                      nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(),
                      layout=model.detecting_head.layout())
assert p2["det"] == 0.0
print("6. batch with no annotated frame     OK")

# ---- 7. downstream helpers work on the flat [M, 5] output ----------------
boxes = decode_predictions(det[0].detach(), conf_thresh=0.0)
assert len(boxes) == M, len(boxes)
best = extract_best_pred_box(det[0].detach())
assert best.shape == (4,)
gtb = extract_gt_box(target[0])
iou = compute_intersection_over_union(best, gtb)
assert 0.0 <= iou.item() <= 1.0
print(f"7. decode/extract on [M, 5], IoU {iou.item():.3f}  OK")

# ---- 8. checkpoint round-trip keeps the anchors --------------------------
ck = {"model_state": model.state_dict(), "anchors": model.anchors_list, "arch": ARCH}
ck_path = os.path.join(tempfile.gettempdir(), "ck.pth")
th.save(ck, ck_path)
ck2 = th.load(ck_path, map_location="cpu", weights_only=False)
m2 = load_handgesture_net(ck2, in_channels=4, n_classes=10)
assert m2.anchors_list == model.anchors_list
m2.eval(); model.eval()
with th.no_grad():
    assert th.allclose(m2(x)[0], model(x)[0], atol=1e-6)
print("8. checkpoint round-trip             OK")

# ---- 9. a checkpoint from another architecture must fail LOUDLY ----------
for bad in ({"model_state": {}}, {"model_state": {}, "arch": "yolov2"}):
    try:
        load_handgesture_net(bad, in_channels=4)
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert ARCH in str(e)
print("9. foreign checkpoint rejected loudly OK")

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
