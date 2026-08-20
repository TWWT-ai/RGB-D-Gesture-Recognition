import torch as th

S = 7

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
#   mean best-IoU, in-sample                 0.8049
#   mean best-IoU, out-of-bag                0.7973 +/- 0.0043   <- honest number
#   aspect-aware hand-picked baseline (k=5)  0.7761
#   -> k-means is worth +0.0288 over a baseline that already knows hands are
#      taller than they are wide. (The +0.16 over SQUARE anchors is a straw man.)
#
# If you change --k or the split, RE-RUN the script and repaste. The numbers
# above are the claim; do not let them drift from the list below.
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
    Inverse of the model output: [S, S, B*5] (one image, after permute)
    -> list of (x_min, y_min, x_max, y_max, conf), all in [0, 1].
    Used by visualize.py / evaluation, NOT by training.
    """
    # >>> [YOLOv2-CHANGED] B is now the number of ANCHORS, which changed from 2
    # to len(ANCHORS). Infer it from the tensor instead of hard-coding it, so
    # changing the anchor list can never silently mis-reshape the output.
    if B is None:
        B = prediction.shape[-1] // 5
    prediction = prediction.view(s, s, B, 5)
    boxes = []
    for row in range(s):
        for col in range(s):
            for b in range(B):
                cx, cy, w, h, conf = prediction[row, col, b].tolist()
                if conf < conf_thresh:
                    continue
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
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
    From the raw head output of ONE image [S, S, B*5],
    return the (cx, cy, w, h) of the box with the highest confidence.
    Center format on purpose, so compute_intersection_over_union can
    compare it with the ground truth directly.
    """
    # >>> [YOLOv2-CHANGED] same reason as decode_predictions: infer the number
    # of anchors from the tensor rather than assuming 2.
    if B is None:
        B = det_pred.shape[-1] // 5
    det_pred = det_pred.view(S, S, B, 5)
    conf = det_pred[..., 4]
    # argmax over the flattened S*S*B boxes, then unravel back to 3 indices
    flat_idx = conf.flatten().argmax()
    row = flat_idx // (S * B)
    col = (flat_idx % (S * B)) // B
    b = flat_idx % B
    return det_pred[row, col, b, :4]