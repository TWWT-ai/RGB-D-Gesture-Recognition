import torch as th

# Grid of the dataloader's TARGET container (encode_yolo_target). Since YOLOv3
# the network predicts on several grids (see GRIDS), but the dataset still
# stores its one ground-truth box in a 7x7 container; the loss reads the box
# back out of it and re-assigns it to whichever grid/anchor fits best.
S = 7

# >>> [YOLOv3-NEW] Prediction grids, coarse -> fine. 448 / 7 = 64 px cells
# (big hands), 448 / 14 = 32 px, 448 / 28 = 16 px (small / far hands).
# YOLOv3 predicts at three scales (13, 26, 52 on a 416 input = strides 32, 16, 8).
# Our backbone already has features at strides 64, 32 and 16 (f7, f14, f28),
# so we use those: the SegmentationHead reads the same tensors, so no extra
# backbone stages are needed.
GRIDS = (7, 14, 28)

# >>> [YOLOv2-NEW] Anchor priors: (width, height) normalised to the image, so
# they live in exactly the same [0, 1] space as encode_yolo_target's w, h.
# YOLOv1 had no anchors at all: each cell regressed B boxes from scratch.
# YOLOv2's "dimension clusters" replace that guesswork with shapes measured
# from the data, so the network only has to predict a small correction.
#
# Produced by:
#   python src/kmeans_anchors.py \
#       --data-root ../data/datasets/dataset_merged_all --seed 42 --k 5
# on the 7 TRAIN persons only (699 annotated boxes). val/test persons were
# excluded, otherwise the priors would be fitted on the people we test on.
#
# !! These were fitted with the DEFAULT --test-frac 0.1, which holds out 2 people
# !! and trains on 7. The Kaggle training command uses --test-frac 0, which holds
# !! out 1 and trains on 8 -- a different split, so this list does NOT match it.
# !! For that run, pass train.py --anchors <json from kmeans_anchors --out-json>;
# !! the JSON records the split it was fitted with and train.py refuses to run if
# !! it disagrees with the split you are training on. This list is only the
# !! fallback for a --test-frac 0.1 run.
#
#   mean best-IoU, in-sample                 0.8049
#   mean best-IoU, out-of-bag                0.7973 +/- 0.0043   <- honest number
#   aspect-aware hand-picked baseline (k=5)  0.7761
#   -> k-means is worth +0.0288 over a baseline that already knows hands are
#      taller than they are wide. (The +0.16 over SQUARE anchors is a straw man.)
#
# If you change --k or the split, RE-RUN the script and repaste. The numbers
# above are the claim; do not let them drift from the list below.
#
# >>> [YOLOv3-NOTE] YOLOv3 uses 9 anchors, 3 per scale. These 5 are still the
# k=5 YOLOv2 clusters, because they are the only ones that were actually
# measured on the data -- inventing 4 more would be making numbers up.
# split_anchors_by_scale() spreads whatever list it is given over GRIDS, so
# 5 anchors run as 2 / 2 / 1 (7x7 / 14x14 / 28x28). For the real YOLOv3
# setting run:
#   python src/kmeans_anchors.py --data-root ... --k 9 --out-json anchors9.json
# and train with --anchors anchors9.json (3 / 3 / 3).
ANCHORS = [
    (0.0826, 0.1763),
    (0.1429, 0.2388),
    (0.1585, 0.3237),
    (0.2143, 0.2656),
    (0.2210, 0.3929),
]


# >>> [YOLOv2-NEW]
def anchor_tensor(anchors=None, device=None, dtype=th.float32):
    """(A, 2) tensor of anchor (w, h). One place, so nothing drifts."""
    if anchors is None:
        anchors = ANCHORS
    return th.tensor([tuple(a) for a in anchors], dtype=dtype, device=device)


# >>> [YOLOv3-NEW]
def split_anchors_by_scale(anchors, n_scales=len(GRIDS)):
    """
    YOLOv3's anchor-to-scale assignment: sort the anchors by AREA, give the
    largest ones to the coarsest grid and the smallest ones to the finest.

    A 7x7 cell is 64 px wide and sees a big receptive field, so it should
    predict big boxes; a 28x28 cell is 16 px and should predict small ones.
    YOLOv3 hard-codes this with its "mask = 6,7,8 / 3,4,5 / 0,1,2" lines;
    here it is derived from the list so changing --k can never mis-assign.

    Returns a list of n_scales lists of (w, h), coarse grid first (same order
    as GRIDS). With 9 anchors that is 3 / 3 / 3; with 5 it is 2 / 2 / 1.
    """
    anchors = [tuple(float(v) for v in a) for a in anchors]
    if len(anchors) < n_scales:
        raise ValueError(f"need at least {n_scales} anchors for {n_scales} scales, "
                         f"got {len(anchors)}")
    ordered = sorted(anchors, key=lambda a: a[0] * a[1], reverse=True)
    base, extra = divmod(len(ordered), n_scales)
    groups, start = [], 0
    for i in range(n_scales):
        n = base + (1 if i < extra else 0)
        groups.append(ordered[start:start + n])
        start += n
    return groups


# >>> [YOLOv2-NEW]
def shape_iou(boxes_wh, anchors_wh):
    """
    IoU between (M,2) ground-truth (w,h) and (A,2) anchor (w,h), pretending
    both are centred on the same point.

    This is SHAPE-only IoU, used to decide which anchor is "responsible" for a
    ground-truth box. YOLOv1 decided that by asking which of its B predictions
    happened to overlap best, which is circular: at the start of training the
    predictions are random, so the assignment is random too. Matching on shape
    is stable from epoch 1.
    """
    inter = (th.minimum(boxes_wh[:, None, 0], anchors_wh[None, :, 0]) *
             th.minimum(boxes_wh[:, None, 1], anchors_wh[None, :, 1]))
    area_b = (boxes_wh[:, 0] * boxes_wh[:, 1])[:, None]
    area_a = (anchors_wh[:, 0] * anchors_wh[:, 1])[None, :]
    return inter / (area_b + area_a - inter + 1e-9)


def compute_intersection_over_union(box1, box2):
    # Computing left and right boundary for both boxes (cx, cy, w, h)
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2         # left bound
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2         # right bound
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    # Calculating intersection for height and width
    inter_w = (th.min(b1_x2, b2_x2) - th.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (th.min(b1_y2, b2_y2) - th.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    # Calculating union
    union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - inter
    return inter / (union + 1e-6)


def encode_yolo_target(boundary_box, s=S):
    x_min, y_min, x_max, y_max = [float(v) for v in boundary_box]
 
    # Corner format -> centre format
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # Which grid cell does the centre fall into?
    # cx * s is in [0, s); int() floors it; min() guards the cx == 1.0 edge case
    col = min(int(cx * s), s - 1)     # column index <- x direction
    row = min(int(cy * s), s - 1)     # row index    <- y direction
    
    target = th.zeros(s, s, 5)
    target[row, col] = th.tensor([cx, cy, w, h, 1.0])
    return target


def decode_predictions(prediction, B=None, conf_thresh=0.25, s=S):
    """
    Inverse of the model output for ONE image -> list of
    (x_min, y_min, x_max, y_max, conf), all in [0, 1].
    Used by visualize.py / evaluation, NOT by training.

    >>> [YOLOv3-CHANGED] the model output is now [M, 5]: every box from every
    grid (7x7, 14x14, 28x28) and every anchor, flattened into one list. The
    boxes are already decoded by the head, so where a box came from no longer
    matters here. reshape(-1, 5) also still accepts the old [S, S, B*5] layout,
    which is why `B` and `s` are kept but ignored.
    """
    prediction = prediction.reshape(-1, 5)
    keep = prediction[:, 4] >= conf_thresh
    boxes = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf)
             for cx, cy, w, h, conf in prediction[keep].tolist()]
    # Highest confidence first
    boxes.sort(key=lambda box: box[4], reverse=True)
    return boxes

def extract_gt_box(det_target):
    """
    Inverse of encode_yolo_target for ONE image:
    [S, S, 5] grid -> the single (cx, cy, w, h) stored in the object cell.
    """
    object_mask = det_target[..., 4] > 0.5
    return det_target[object_mask][0, :4]


def extract_best_pred_box(det_pred, B=None):
    """
    From the head output of ONE image, return the (cx, cy, w, h) of the box
    with the highest confidence. Center format on purpose, so
    compute_intersection_over_union can compare it with the ground truth
    directly.

    >>> [YOLOv3-CHANGED] det_pred is [M, 5] across all three grids now. The
    argmax runs over every box at every scale, so a big hand found on the 7x7
    grid and a small one found on the 28x28 grid compete on confidence alone.
    """
    det_pred = det_pred.reshape(-1, 5)
    return det_pred[det_pred[:, 4].argmax(), :4]