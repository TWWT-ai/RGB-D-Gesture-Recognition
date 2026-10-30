import torch as th
import math

# Grid of the dataloader's TARGET container (encode_yolo_target). Since YOLOv3
# the network predicts on several grids (see GRIDS), but the dataset still
# stores its one ground-truth box in a 7x7 container; the loss reads the box
# back out of it and re-assigns it to whichever grid points fit best.
S = 7

# >>> [YOLOv3-NEW] Prediction grids, coarse -> fine. 448 / 7 = 64 px cells
# (big hands), 448 / 14 = 32 px, 448 / 28 = 16 px (small / far hands).
# YOLOv3 predicts at three scales (13, 26, 52 on a 416 input = strides 32, 16, 8).
# Our backbone already has features at strides 64, 32 and 16 (f7, f14, f28),
# so we use those: the SegmentationHead reads the same tensors, so no extra
# backbone stages are needed.
GRIDS = (7, 14, 28)

# >>> [YOLOv10-CHANGED] The anchor machinery that lived here since YOLOv2
# (ANCHORS, anchor_tensor, shape_iou, split_anchors_by_scale and YOLOv5's
# anchor_ratio_metric / anchor_fitness) is gone: YOLOv10 is anchor-free, every
# grid-cell centre predicts its distances to the four box sides directly.
# The k=5 clusters measured for YOLOv2 were
#   (0.0826, 0.1763) (0.1429, 0.2388) (0.1585, 0.3237) (0.2143, 0.2656) (0.2210, 0.3929)
# and remain on the yolov2..yolov5 branches.


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


# >>> [YOLOv4-NEW]
def bbox_ciou(box1, box2, eps=1e-7):
    """
    Complete-IoU between boxes in (cx, cy, w, h) format, broadcastable.

        CIoU = IoU - rho^2 / c^2 - alpha * v

    IoU      overlap, as before
    rho^2/c^2  squared centre distance over the squared diagonal of the
             smallest box enclosing both -> still gives a gradient when the
             boxes do NOT overlap at all, where plain IoU is flat zero
    v        how different the two aspect ratios are,
             (4 / pi^2) * (atan(w_gt / h_gt) - atan(w / h))^2
    alpha    v / ((1 - IoU) + v), so the aspect term only takes over once the
             boxes already overlap

    YOLOv4 trains the box with loss = 1 - CIoU instead of MSE on x, y, w, h.
    MSE treats the four numbers as independent, but the metric we are judged
    on (IoU) is not: a 5 px error is huge on a small hand and nothing on a big
    one. CIoU optimises the overlap directly and is scale-invariant.
    Range: [-1, 1], 1 = identical boxes.
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    inter = ((th.min(b1_x2, b2_x2) - th.max(b1_x1, b2_x1)).clamp(min=0) *
             (th.min(b1_y2, b2_y2) - th.max(b1_y1, b2_y1)).clamp(min=0))
    w1, h1 = box1[..., 2], box1[..., 3] + eps
    w2, h2 = box2[..., 2], box2[..., 3] + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # smallest enclosing box
    cw = th.max(b1_x2, b2_x2) - th.min(b1_x1, b2_x1)
    ch = th.max(b1_y2, b2_y2) - th.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (box1[..., 0] - box2[..., 0]) ** 2 + (box1[..., 1] - box2[..., 1]) ** 2

    v = (4 / math.pi ** 2) * (th.atan(w2 / h2) - th.atan(w1 / h1)) ** 2
    with th.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


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
    # >>> [YOLOv10-CHANGED] the head output carries the raw one-to-many /
    # one-to-one logits after the first 5 channels; only the first 5 (the
    # decoded one-to-one boxes) are boxes. Thanks to the one-to-one head
    # these need NO non-maximum suppression: it is trained to give each hand
    # exactly one high score.
    prediction = prediction.reshape(-1, prediction.shape[-1])[:, :5]
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
    # >>> [YOLOv10-CHANGED] first 5 channels = decoded one-to-one boxes
    det_pred = det_pred.reshape(-1, det_pred.shape[-1])[:, :5]
    return det_pred[det_pred[:, 4].argmax(), :4]