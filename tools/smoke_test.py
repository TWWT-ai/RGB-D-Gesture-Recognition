"""CPU smoke test for the YOLOv10 port. Run from src/: python ../tools/smoke_test.py"""
import sys, os, math, tempfile, copy
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "src"), os.path.join(_here, "..", "src")):
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    raise SystemExit("cannot find src/ next to or above this script")

import torch as th
import numpy as np

from utils import (GRIDS, encode_yolo_target, decode_predictions, extract_best_pred_box,
                   extract_gt_box, compute_intersection_over_union, S, bbox_ciou)
from model import (HandGestureNet, DetectionHead, load_handgesture_net, ARCH, REG_MAX,
                   SPPF, C2f, SCDown, CIB, C2fCIB, PSA)
from train import (yolo_detection_loss, compute_loss, gt_boxes_from_target, task_aligned_assign,
                   head_loss, dfl_loss, ModelEMA, param_groups, TOPK_O2M, TOPK_O2O)
import torch.nn as nn

th.manual_seed(0)
M = sum(g * g for g in GRIDS)
K = 4 * REG_MAX + 1
C = 5 + 2 * K
print(f"anchor-free: {M} points over grids {GRIDS}, head output [N, {M}, {C}]")

# ---- 0. building blocks keep / change shapes as intended -----------------
x14 = th.randn(1, 64, 14, 14)
assert C2f(64, 96, n=2)(x14).shape == (1, 96, 14, 14)
assert SCDown(64, 128)(x14).shape == (1, 128, 7, 7)
assert CIB(64, 64, lk=True)(x14).shape == (1, 64, 14, 14)
assert C2fCIB(64, 64, n=1, lk=True)(x14).shape == (1, 64, 14, 14)
assert PSA(128)(th.randn(1, 128, 7, 7)).shape == (1, 128, 7, 7)
sppf = SPPF(8, 8).eval()
x7 = th.randn(1, 8, 7, 7)
with th.no_grad():
    h = sppf.cv1(x7)
    spp = th.cat([h] + [nn.functional.max_pool2d(h, k, 1, k // 2) for k in (5, 9, 13)], 1)
    assert th.allclose(sppf.cv2(spp), sppf(x7), atol=1e-6)
# SCDown really is cheaper than a dense stride-2 3x3
dense = 3 * 3 * 512 * 1024
assert sum(p_.numel() for p_ in SCDown(512, 1024).parameters()) < dense / 4
print("0. C2f / SCDown / CIB / C2fCIB / PSA / SPPF shapes  OK")

# ---- 1. forward shape + ranges -------------------------------------------
model = HandGestureNet(in_channels=4, n_classes=10)
x = th.randn(2, 4, 448, 448)
det, seg, cls = model(x)
assert det.shape == (2, M, C), det.shape
assert seg.shape == (2, 1, 448, 448), seg.shape
assert cls.shape == (2, 10), cls.shape
assert th.isfinite(det).all()
d = det[..., :5]
assert (d[..., 4] >= 0).all() and (d[..., 4] <= 1).all(), "score out of [0,1]"
assert (d[..., 2] > 0).all() and (d[..., 3] > 0).all(), "w/h must be positive"
assert d[..., 4].mean() < 0.2, d[..., 4].mean()        # score prior log(5 / S^2)
print("1. forward shapes + ranges           OK")

# ---- 2. anchor-free decode: every box contains its own point -------------
head = model.detecting_head
pts, grid = head.points, head.point_grid
assert pts.shape == (M, 2) and grid.shape == (M,)
x1 = d[..., 0] - d[..., 2] / 2; x2 = d[..., 0] + d[..., 2] / 2
y1 = d[..., 1] - d[..., 3] / 2; y2 = d[..., 1] + d[..., 3] / 2
assert ((x1 <= pts[:, 0] + 1e-5) & (pts[:, 0] <= x2 + 1e-5) &
        (y1 <= pts[:, 1] + 1e-5) & (pts[:, 1] <= y2 + 1e-5)).all()
# and each side is at most REG_MAX - 1 cells of its own grid away
assert ((pts[:, 0] - x1) * grid <= REG_MAX - 1 + 1e-4).all()
print("2. ltrb decode around each point, <= reg_max-1 cells  OK")

# ---- 3. task-aligned assigner --------------------------------------------
gt = th.tensor([[0.43, 0.52, 0.2, 0.3, 1.0]])
pred = th.tensor([0.43, 0.52, 0.2, 0.3]).expand(1, M, 4).clone()
scores = th.rand(1, M)
t_m, pos_m = task_aligned_assign(pred, scores, gt, gt[:, 4] > 0.5, pts, TOPK_O2M)
t_o, pos_o = task_aligned_assign(pred, scores, gt, gt[:, 4] > 0.5, pts, TOPK_O2O)
assert int(pos_m.sum()) == TOPK_O2M and int(pos_o.sum()) == TOPK_O2O
# consistent matching metric: the one-to-one pick is one of the one-to-many picks
assert (pos_o & pos_m).sum() == 1
# positives lie inside the box; targets are 0 off the positives, <= 1 on them
pp = pts[pos_m[0]]
assert ((pp[:, 0] > 0.33) & (pp[:, 0] < 0.53) & (pp[:, 1] > 0.37) & (pp[:, 1] < 0.67)).all()
assert (t_m[~pos_m] == 0).all() and (t_m[pos_m] > 0).all() and t_m.max() <= 1 + 1e-6
print(f"3. TAL: o2m {int(pos_m.sum())} / o2o {int(pos_o.sum())} positives, o2o in o2m, inside box  OK")

# ---- 4. loss is finite, gradients flow; o2o head does NOT train the backbone
target = th.zeros(2, S, S, 5)
target[0] = encode_yolo_target([0.30, 0.35, 0.55, 0.70])   # x1,y1,x2,y2
target[1] = encode_yolo_target([0.10, 0.10, 0.25, 0.30])
loss = yolo_detection_loss(det, target, head.layout())
assert th.isfinite(loss), loss
loss.backward()
gnorm = sum(p_.grad.abs().sum().item() for p_ in model.parameters() if p_.grad is not None)
assert gnorm > 0 and math.isfinite(gnorm)
model.zero_grad()
det2 = model(x)[0]
lay = head.layout()
gt2 = gt_boxes_from_target(target)
o2o_only = head_loss(det2[..., 5 + K:5 + 2 * K], gt2, gt2[:, 4] > 0.5, lay, TOPK_O2O)
o2o_only.backward()
bb = sum(p_.grad.abs().sum().item() for p_ in model.backbone.parameters() if p_.grad is not None)
o2o_g = sum(p_.grad.abs().sum().item() for p_ in head.o2o_box.parameters() if p_.grad is not None)
assert bb == 0 and o2o_g > 0, (bb, o2o_g)
model.zero_grad()
print(f"4. det loss {loss.item():.3f}; o2o loss reaches its head ({o2o_g:.1f}) but not the backbone  OK")

# ---- 4b. DFL + a perfect prediction gives ~zero box / dfl loss -----------
t = th.tensor([[2.3, 0.0, 14.99, 7.0]])
logits = th.full((1, 4, REG_MAX), -30.0)
for i, v in enumerate(t[0].tolist()):
    lo = min(int(v), REG_MAX - 2); w_lo = lo + 1 - min(v, REG_MAX - 1 - 0.01)
    logits[0, i, lo] = math.log(max(w_lo, 1e-12)); logits[0, i, lo + 1] = math.log(max(1 - w_lo, 1e-12))
# the ideal distribution has DFL == its own entropy, and a shifted one is worse
ideal = dfl_loss(logits, t, REG_MAX).item()
worse = dfl_loss(logits.roll(1, dims=-1), t, REG_MAX).item()
assert worse > ideal + 1.0, (ideal, worse)
assert abs((logits.softmax(-1) @ th.arange(REG_MAX).float())[0, 0].item() - 2.3) < 1e-3
print(f"4b. DFL: ideal {ideal:.3f} < shifted {worse:.3f}, mean of bins == distance  OK")

# ---- 4c. CIoU ------------------------------------------------------------
b1 = th.tensor([0.5, 0.5, 0.2, 0.3])
assert abs(bbox_ciou(b1, b1).item() - 1.0) < 1e-5
far = th.tensor([0.9, 0.9, 0.05, 0.05], requires_grad=True)
c = bbox_ciou(far, b1)
assert c.item() < 0 and compute_intersection_over_union(far, b1).item() == 0
c.backward()
assert far.grad.abs().sum() > 0
print("4c. CIoU                              OK")

# ---- 4d. EMA, param groups -----------------------------------------------
shadow = copy.deepcopy(model)            # do not disturb the shared model, later tests use it
ema = ModelEMA(shadow)
w0 = next(ema.ema.parameters()).clone()
with th.no_grad():
    for p_ in shadow.parameters():
        p_.add_(1.0)
ema.update(shadow)
assert ema.decay(1) < 0.01 and (next(ema.ema.parameters()) - w0).abs().mean() > 0.99
groups = param_groups(model, 1e-4)
assert sum(p_.numel() for g_ in groups for p_ in g_["params"]) == sum(p_.numel() for p_ in model.parameters())
print("4d. EMA ramp, no-decay groups         OK")

# ---- 5. multi-task compute_loss path -------------------------------------
det, seg, cls = model(x)
targets = {"det": target, "Mask": th.zeros(2, 1, 448, 448),
           "Label": th.tensor([1, 2]), "has_mask": th.tensor([1.0, 0.0])}
total, parts = compute_loss((det, seg, cls), targets, nn.BCEWithLogitsLoss(),
                            nn.CrossEntropyLoss(), layout=head.layout())
assert all(math.isfinite(v) for v in parts.values()), parts
total.backward()
print(f"5. compute_loss {parts}   OK")

# ---- 6. all-unannotated batch must not crash -----------------------------
targets["has_mask"] = th.tensor([0.0, 0.0])
t2, p2 = compute_loss((det.detach(), seg.detach(), cls.detach()), targets,
                      nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), layout=head.layout())
assert p2["det"] == 0.0
print("6. batch with no annotated frame     OK")

# ---- 7. downstream helpers read only the 5 decoded o2o channels ----------
boxes = decode_predictions(det[0].detach(), conf_thresh=0.0)
assert len(boxes) == M, len(boxes)
best = extract_best_pred_box(det[0].detach())
assert best.shape == (4,)
assert th.equal(best, det[0, det[0, :, 4].argmax(), :4].detach())
iou = compute_intersection_over_union(best, extract_gt_box(target[0]))
assert 0.0 <= iou.item() <= 1.0
print(f"7. decode/extract on [M, {C}], IoU {iou.item():.3f}  OK")

# ---- 8. checkpoint round-trip --------------------------------------------
ck = {"model_state": model.state_dict(), "arch": ARCH}
ck_path = os.path.join(tempfile.gettempdir(), "ck.pth")
th.save(ck, ck_path)
ck2 = th.load(ck_path, map_location="cpu", weights_only=False)
m2 = load_handgesture_net(ck2, in_channels=4, n_classes=10)
m2.eval(); model.eval()
with th.no_grad():
    assert th.allclose(m2(x)[0], model(x)[0], atol=1e-5)
print("8. checkpoint round-trip             OK")

# ---- 9. a checkpoint from another architecture must fail LOUDLY ----------
for bad in ({"model_state": {}}, {"model_state": {}, "arch": "yolov5"}):
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
