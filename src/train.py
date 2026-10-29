import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import math     # >>> [YOLOv5-NEW]
import copy     # >>> [YOLOv5-NEW]
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataloader import get_dataLoaders
from dataloader import HandGestureDataset
from model import HandGestureNet, ARCH     # >>> [YOLOv3-CHANGED]
from utils import compute_intersection_over_union
from utils import shape_iou, anchor_tensor, ANCHORS     # >>> [YOLOv2-NEW]
from utils import bbox_ciou     # >>> [YOLOv4-NEW]
from utils import anchor_ratio_metric, anchor_fitness     # >>> [YOLOv5-NEW]
# Grid size
S = 7

#---------------------------------------------------
# Loss Function
#---------------------------------------------------

def gt_boxes_from_target(target):
    """
    >>> [YOLOv3-NEW] The dataset still stores its single box in the old 7x7
    container (encode_yolo_target). Pull the (cx, cy, w, h, has_box) row back
    out, one per image: [N, S, S, 5] -> [N, 5].
    """
    flat = target.reshape(target.size(0), -1, 5)
    idx = flat[..., 4].argmax(dim=1)
    return flat[th.arange(flat.size(0), device=flat.device), idx]


def _bce(p, y, eps=1e-7):
    # BCE on a probability. The head hands out sigmoid(t), not t, so this is
    # written out by hand with an eps instead of clamping p: a clamp would zero
    # the gradient of a positive that is stuck near p = 0, which is exactly the
    # case that most needs one.
    return -(y * th.log(p + eps) + (1 - y) * th.log(1 - p + eps))


def build_targets(gt, layout, anchor_t=4.0):
    """
    >>> [YOLOv5-CHANGED] replaces YOLOv4's assign_targets. Which
    (image, anchor, cell) predictions are POSITIVES?

    1. Anchor match by SIZE RATIO, not shape-IoU: an anchor is used if the box
       is within anchor_t (4x) of it in both width and height
       (utils.anchor_ratio_metric). This is exactly the range the new decode
       can reach, so no anchor is ever asked for a box it cannot produce.
       (YOLOv5 drops a box that no anchor matches; with ONE hand per image
       that would lose the whole frame, so the closest anchor is always kept.)
    2. Neighbouring cells: besides the cell that holds the centre, the TWO
       nearest neighbouring cells (one horizontal, one vertical -- the side
       of the cell the centre is closer to) are positives too. The new decode
       lets a cell reach centres in (col - 0.5, col + 1.5), so they can.
       That is up to 3x more positives per anchor, which is the main reason
       YOLOv5 converges faster.

    gt: [n, 4] (cx, cy, w, h). Returns (box_idx, flat) per positive.
    """
    anchors = layout["anchors"].to(gt.dtype)
    ratio = anchor_ratio_metric(gt[:, 2:4], anchors)              # [n, A]
    match = ratio > 1.0 / anchor_t
    match[th.arange(gt.size(0), device=gt.device), ratio.argmax(dim=1)] = True
    box_idx, anchor = th.nonzero(match, as_tuple=True)
    g = gt[box_idx]
    S_a = layout["grid"][anchor]
    Sf = S_a.to(gt.dtype)

    gxy = g[:, :2] * Sf[:, None]                  # centre in grid units
    gxi = Sf[:, None] - gxy                       # same, measured from the far side
    left_up = (gxy % 1 < 0.5) & (gxy > 1)         # closer to the left / top border
    right_down = (gxi % 1 < 0.5) & (gxi > 1)      # closer to the right / bottom border
    #           centre,  left,          up,            right,           down
    use = th.stack([th.ones_like(left_up[:, 0]), left_up[:, 0], left_up[:, 1],
                    right_down[:, 0], right_down[:, 1]])           # [5, n]
    off = th.tensor([[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]], device=gt.device)

    k, j = th.nonzero(use, as_tuple=True)        # (which offset, which match)
    col = th.minimum(gxy[j, 0].long(), S_a[j] - 1) + off[k, 0]
    row = th.minimum(gxy[j, 1].long(), S_a[j] - 1) + off[k, 1]
    col = th.maximum(th.minimum(col, S_a[j] - 1), th.zeros_like(col))
    row = th.maximum(th.minimum(row, S_a[j] - 1), th.zeros_like(row))
    a = anchor[j]
    flat = layout["offset"][a] + (row * S_a[j] + col) * layout["count"][a] + layout["local"][a]
    return box_idx[j], flat


# >>> [YOLOv5-NEW] YOLOv5's hyper-parameters (hyp.scratch-low.yaml), already
# rescaled the way train.py in YOLOv5 rescales them: box *= 3 / n_layers,
# obj *= (img_size / 640)^2 * 3 / n_layers  ->  3 layers, 448 px.
BOX_GAIN = 0.05
OBJ_GAIN = 1.0 * (448 / 640) ** 2
# Objectness weight per grid: YOLOv5's [4.0, 1.0, 0.4] for P3 / P4 / P5. The
# finest grid has 16x more cells than the coarsest, so its MEAN objectness
# loss is much smaller per cell; the balance gives each scale a fair say.
OBJ_BALANCE = {28: 4.0, 14: 1.0, 7: 0.4}
# YOLOv5 multiplies its loss by the batch size. Here that would put train
# (~3 annotated frames per batch) and val (16 per batch) on different scales,
# so the default batch size is used as a fixed constant instead.
LOSS_SCALE = 16.0


def yolo_detection_loss(prediction, target, layout, anchor_t=4.0,
                        box_gain=BOX_GAIN, obj_gain=OBJ_GAIN, loss_scale=LOSS_SCALE):
    """
    >>> [YOLOv5-CHANGED] (on top of the YOLOv4 version)

        Steps:
        1. Read the one ground-truth box per image out of the target ->
        2. build_targets(): ratio-matched anchors x (centre + 2 neighbour cells) ->
        3. Box term = mean(1 - CIoU) over the positives ->
        4. Objectness: BCE per grid against an IoU-AWARE target, averaged per
           grid and weighted by OBJ_BALANCE ->
        5. box_gain * box + obj_gain * obj

    What changed from YOLOv4:
    - Positives: ratio match + neighbour cells (see build_targets).
    - Objectness target for a positive is its own CIoU with the truth
      (clamped at 0), not 1. A badly placed positive box is no longer
      trained to be confident, so confidence ranks boxes by how good they
      are -- which is exactly what extract_best_pred_box relies on.
    - No ignore threshold: with an IoU-valued target it is not needed.
    - Losses are MEANS (box over positives, objectness over every cell of a
      grid) with YOLOv5's gains, instead of per-image sums.
    """
    N, M, _ = prediction.shape
    gt = gt_boxes_from_target(target)                     # [N, 5]
    has_box = gt[:, 4] > 0.5
    conf = prediction[..., 4]                             # [N, M]
    obj_target = th.zeros_like(conf)

    loss_box = prediction.new_zeros(())
    if has_box.any():
        idx = th.nonzero(has_box).squeeze(1)
        box_idx, flat = build_targets(gt[idx, :4], layout, anchor_t)
        img = idx[box_idx]
        response = prediction[img, flat]                  # [n_pos, 5]
        ciou = bbox_ciou(response[:, :4], gt[img, :4])
        loss_box = (1.0 - ciou).mean()
        obj_target[img, flat] = ciou.detach().clamp(min=0).to(obj_target.dtype)

    loss_obj = prediction.new_zeros(())
    for S_i, start, end in layout["scales"]:
        loss_obj = loss_obj + OBJ_BALANCE.get(S_i, 1.0) * _bce(conf[:, start:end],
                                                                obj_target[:, start:end]).mean()

    # Returning Multi Part loss function
    return (box_gain * loss_box + obj_gain * loss_obj) * loss_scale


def compute_loss(outputs, targets, segmentation_criterion, classification_criterion, lambda_seg=1.0, lambda_cls=1.0, lambda_det=1.0,
                 layout=None):     # >>> [YOLOv3-CHANGED] anchors -> full multi-scale layout
    # Computing loss function for each Head defined in models
    detection_pred, segmentation_pred, classification_pred = outputs

    # Classification is trained on EVERY sample (annotated or not) -> uses all data
    loss_classification = classification_criterion(classification_pred, targets["Label"])

    # Detection + segmentation only make sense where we actually have a mask/box.
    # has_mask is a per-sample flag (1.0 = annotated, 0.0 = classification-only).
    has_mask = targets["has_mask"].bool()
    if has_mask.any():
        loss_detection = yolo_detection_loss(detection_pred[has_mask], targets["det"][has_mask],
                                             layout)     # >>> [YOLOv3-CHANGED]
        loss_segmentation = segmentation_criterion(segmentation_pred[has_mask],
                                                   targets["Mask"][has_mask].float())
    else:
        # No annotated sample in this batch -> nothing to learn for det/seg here
        zero = classification_pred.new_zeros(())
        loss_detection = zero
        loss_segmentation = zero

    total = lambda_det * loss_detection + lambda_seg * loss_segmentation + lambda_cls * loss_classification
    # DETACHED TENSORS, not floats. float(x) on a CUDA tensor forces a
    # device->host sync, and this ran four times per batch purely for logging --
    # which stalls the pipeline so the CPU can never run ahead and queue the next
    # batch. The callers accumulate these on-device and .item() once per epoch.
    losses = {"det": loss_detection.detach(),
              "seg": loss_segmentation.detach(),
              "cls": loss_classification.detach(),
              "total": total.detach()
              }
    return total, losses


#---------------------------------------------------
# >>> [YOLOv5-NEW] Training machinery from YOLOv5
#---------------------------------------------------

class ModelEMA:
    """
    Exponential moving average of the weights (YOLOv5's ModelEMA).

    After every optimizer step:  ema = d * ema + (1 - d) * weights,
    with d ramping up as 0.9999 * (1 - exp(-updates / 2000)) so the average
    does not drag in the random initial weights. Validation, checkpointing and
    therefore evaluate.py all use the EMA copy: it is a free ensemble of the
    last few thousand steps and noticeably less noisy from epoch to epoch,
    which also makes the best.pth guardrail below more trustworthy.
    BatchNorm running statistics are averaged the same way.
    """
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @th.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def param_groups(model, weight_decay):
    """
    YOLOv5's three parameter groups: weight decay ONLY on conv / linear
    weights. Decaying BatchNorm scales and biases just pulls them towards 0,
    which fights the normalisation instead of regularising anything.
    """
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "weight" and not isinstance(module, nn.modules.batchnorm._BatchNorm):
                decay.append(p)
            else:
                no_decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def check_anchors(data_root, train_persons, anchors, anchor_t=4.0, thr=0.98):
    """
    YOLOv5's autoanchor, run once before training: measure how well the
    anchors cover the TRAINING boxes; if fewer than 98% of boxes can be
    reached by some anchor (bpr), re-fit k anchors with IoU k-means and keep
    them only if they are better. The chosen anchors are saved in the
    checkpoint like before, so evaluation always uses the same ones.
    """
    from kmeans_anchors import collect_wh, best_of_n_restarts
    import numpy as np
    wh = collect_wh(data_root, set(train_persons))
    boxes = th.tensor(wh, dtype=th.float32)
    bpr, aat = anchor_fitness(boxes, anchor_tensor(anchors), anchor_t)
    print(f"AutoAnchor: {aat:.2f} anchors/target, best possible recall {bpr:.4f}")
    if bpr >= thr:
        print("AutoAnchor: current anchors fit the data, keeping them")
        return anchors
    new, _ = best_of_n_restarts(np.asarray(wh), len(anchors))
    new = [(round(float(w), 4), round(float(h), 4)) for w, h in new]
    new_bpr, new_aat = anchor_fitness(boxes, anchor_tensor(new), anchor_t)
    print(f"AutoAnchor: re-fitted {len(new)} anchors -> {new_aat:.2f} anchors/target, "
          f"best possible recall {new_bpr:.4f}")
    if new_bpr > bpr:
        print(f"AutoAnchor: using the new anchors {new}")
        return new
    print("AutoAnchor: re-fit was not better, keeping the original anchors")
    return anchors


#---------------------------------------------------
# Train and validation
#---------------------------------------------------

def _forward_fp32(model, images, use_amp):
    """
    Run the network under autocast, then hand the heads back in fp32.

    The convolutions are the part worth running in half precision; the losses are
    not. yolo_detection_loss sums squared errors and divides by an IoU, and the
    detection head's decode goes through exp() -- all of that is happier in fp32,
    and it is a rounding-error slice of the runtime anyway.
    """
    device_type = images.device.type
    with th.autocast(device_type=device_type, dtype=th.float16, enabled=use_amp):
        outputs = model(images)
    return tuple(o.float() for o in outputs)


def train_one_epoch(model, loader, optimizer, segmentation_criterion,
                    classification_criterion, device, lambda_segmentation,
                    lambda_classification, lambda_detection=1.0,
                    scaler=None, use_amp=False,
                    ema=None, warmup=None):     # >>> [YOLOv5-NEW]
    model.train()
    # >>> [YOLOv2-NEW] read the anchors off the model, so the loss can never
    # be matched against a different anchor set from the one the head decoded with
    # >>> [YOLOv3-CHANGED] ...together with which grid each anchor lives on
    layout = model.detecting_head.layout()
    # Accumulate ON DEVICE as tensors -- see the note in compute_loss. These are
    # only pulled back to the host once, after the whole epoch.
    zero = th.zeros((), device=device)
    running = {"det": zero.clone(),
              "seg": zero.clone(),
              "cls": zero.clone(),
              "total": zero.clone()
              }
    correct = zero.clone()                 # track TRAIN classification accuracy
    total_samples = 0

    for i, (images, targets) in enumerate(loader):
        # >>> [YOLOv5-NEW] per-iteration learning-rate warmup (no-op once over)
        if warmup is not None:
            warmup(i)
        # Copying what is on CPU onto GPU (RAM to VRAM). non_blocking pairs with
        # the loader's pin_memory=True so the copy overlaps with compute.
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        # Computing loss for each image
        outputs = _forward_fp32(model, images, use_amp)
        loss, parts = compute_loss(outputs, targets, segmentation_criterion,
                                   classification_criterion, lambda_segmentation,
                                   lambda_classification, lambda_detection,
                                   layout=layout)     # >>> [YOLOv3-CHANGED]

        # Cleaning the previously loaded data first
        optimizer.zero_grad(set_to_none=True)
        # The chain rule part, this tells you on which way you should adjust the direction
        # dloss/dw = dloss/dy * dy/dz * dz/dw
        # Under fp16 the gradients are scaled up before backward and unscaled
        # before the step, so small gradients do not flush to zero.
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Using Adam formula to update the and override the old data
            optimizer.step()
        if ema is not None:
            ema.update(model)     # >>> [YOLOv5-NEW]

        # Adding the loss to the training model
        for k in running:
            running[k] += parts[k]

        # Train classification accuracy (compare against val to spot over/underfitting)
        pred_label = outputs[2].argmax(dim=1)
        correct += (pred_label == targets["Label"]).sum()
        total_samples += images.size(0)

    n = len(loader)
    metrics = {k: v.item() / n for k, v in running.items()}
    metrics["cls_acc"] = correct.item() / total_samples
    return metrics

@th.no_grad()
def validate(model, loader, segmentation_criterion,
            classification_criterion, device, lambda_segmentation,
            lambda_classification, lambda_detection=1.0, use_amp=False):
    model.eval()
    layout = model.detecting_head.layout()     # >>> [YOLOv3-CHANGED]
    zero = th.zeros((), device=device)
    running = {"det": zero.clone(),
              "seg": zero.clone(),
              "cls": zero.clone(),
              "total": zero.clone()
              }
    correct = zero.clone()
    total_samples = 0

    for images, targets in loader:
        # Copying what is on CPU onto GPU (RAM to VRAM)
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        outputs = _forward_fp32(model, images, use_amp)
        _, parts = compute_loss(outputs, targets, segmentation_criterion,
                                   classification_criterion, lambda_segmentation,
                                   lambda_classification, lambda_detection,
                                   layout=layout)     # >>> [YOLOv3-CHANGED]

        # Adding the loss to the training model
        for k in running:
            running[k] += parts[k]

        # Classification accuracy
        pred_label = outputs[2].argmax(dim=1)
        correct += (pred_label == targets["Label"]).sum()
        total_samples += images.size(0)

    n = len(loader)
    metrics = {k: v.item() / n for k, v in running.items()}
    metrics["cls_acc"] = correct.item() / total_samples
    return metrics


#---------------------------------------------------
# Main method
#---------------------------------------------------

def main():
    # Like creating an empty sheet with all the data we are tracking
    parser = argparse.ArgumentParser(description="Train HandGestureNet (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=50)
    # matches get_dataLoaders' own default; the two used to disagree (8 vs 16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-det", type=float, default=1.0)
    # 2 workers could not keep a GPU fed: __getitem__ is ~28 ms of PNG decoding,
    # so the loader, not the network, set the epoch time.
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    # Decode+resize each frame ONCE into a memmap instead of every epoch. Point
    # this at writable scratch (on Kaggle: /kaggle/working/cache, since
    # /kaggle/input is read-only). Empty = no cache, decode every time.
    parser.add_argument("--cache-dir", type=str, default="",
                        help="directory for the decoded-frame cache; "
                             "empty disables caching")
    # Mixed precision. On by default for CUDA, ignored on CPU.
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # Checkpoint rule: cls_acc must improve AND seg/det val losses must stay
    # within (1 + best-tol) of the best they have ever been
    parser.add_argument("--best-tol", type=float, default=0.15)
    # >>> [YOLOv4-NEW] class label smoothing, from YOLOv4's "bag of freebies".
    # Target becomes 0.9 for the true gesture and 0.1/9 for the others, so the
    # classifier is not pushed towards infinitely confident logits on the
    # training people -- a cheap regulariser for the unseen-person test.
    # YOLOv4 applies it to the detector's class scores; here the detector has
    # one class (hand), so it goes on the gesture ClassificationHead instead.
    # --label-smoothing 0 gives back the exact YOLOv3-branch behaviour.
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    # >>> [YOLOv5-NEW] YOLOv5 training defaults
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
                        help="linear LR warmup length (YOLOv5 default 3); 0 disables")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="final LR = lr * lrf at the end of the cosine schedule")
    parser.add_argument("--ema", dest="ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--autoanchor", dest="autoanchor", action="store_true", default=True)
    parser.add_argument("--no-autoanchor", dest="autoanchor", action="store_false")
    parser.add_argument("--out-dir", type=str, default="weights/")
    parser.add_argument("--seed", type=int, default=42)
    # >>> [YOLOv2-NEW] anchors from a file beat anchors hand-pasted into utils.py:
    # on Kaggle/Colab the repo is cloned fresh every run, so editing utils.py by
    # hand is both awkward and easy to get silently out of date.
    parser.add_argument("--anchors", type=str, default="",
                        help="JSON produced by kmeans_anchors.py --out-json; "
                             "empty means use the ANCHORS list in utils.py")
    # Pack all the top argument into one
    args = parser.parse_args()

    # >>> [YOLOv2-NEW] resolve the anchors BEFORE anything expensive, so a
    # mismatched split aborts in a second rather than after scanning the dataset.
    anchors = ANCHORS
    if args.anchors:
        with open(args.anchors, "r", encoding="utf-8") as f:
            meta = json.load(f)
        anchors = [tuple(a) for a in meta["anchors"]]
        # The split the anchors were fitted on MUST match the split we are about
        # to train on, otherwise the priors have seen people this run holds out.
        fitted = meta.get("split", {})
        here = {"seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac}
        if fitted and fitted != here:
            raise SystemExit(
                f"\nAnchors in {args.anchors} were fitted with split {fitted},\n"
                f"but you are training with {here}.\n"
                f"Re-run kmeans_anchors.py with the same --seed/--val-frac/--test-frac.")
        print(f"Anchors loaded from {args.anchors} "
              f"(out-of-bag mean best-IoU {meta.get('mean_best_iou_oob', '?')})")

    # Keeping the training sample set the same every training at the start
    th.manual_seed(args.seed)
    # Selecting the available GPU device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = bool(args.amp) and device.type == "cuda"
    if device.type == "cuda":
        # The input shape is fixed at 448x448, so let cuDNN benchmark its
        # algorithms once on the first batch and reuse the winner after that.
        th.backends.cudnn.benchmark = True
    scaler = th.amp.GradScaler("cuda") if use_amp else None
    print(f"Mixed precision: {'on (fp16)' if use_amp else 'off (fp32)'}  |  "
          f"dataloader workers: {args.num_workers}")

    # Splitting data (3-way by person; test_loader is NEVER touched here —
    # it exists so the split is carved out consistently, evaluate.py uses it)
    train_loader, validation_loader, test_loader = get_dataLoaders(
        args.data_root,
        batch_size=args.batch_size,
        n_val_persons=args.n_val_persons,
        seed=args.seed,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        cache_dir=args.cache_dir or None,
    )
    
    # >>> [YOLOv5-NEW] autoanchor on the TRAIN persons only (same leakage rule
    # as kmeans_anchors.py)
    if args.autoanchor:
        train_persons = {s["Person"] for s in train_loader.dataset.samples}
        anchors = check_anchors(args.data_root, train_persons, anchors)

    #Creating model, optimizer, loss
    model = HandGestureNet(in_channels=4, n_classes=10, anchors=anchors).to(device)
    print(f"Detection head: {len(anchors)} anchors (w, h, normalised) = "
          f"{[(round(a, 3), round(b, 3)) for a, b in anchors]}")
    # >>> [YOLOv3-NEW] show which anchors went to which grid
    for S_i, group in zip(model.detecting_head.grids, model.detecting_head.groups):
        print(f"  {S_i:>2}x{S_i:<2} grid: {[(round(a, 3), round(b, 3)) for a, b in group]}")
    segmentation_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)     # >>> [YOLOv4-CHANGED]
    # >>> [YOLOv5-CHANGED] weight decay only on conv/linear weights (param_groups)
    optimizer = th.optim.Adam(param_groups(model, args.weight_decay), lr=args.lr)
    # >>> [YOLOv5-CHANGED] YOLOv5's cosine "one-cycle" factor, lr -> lr * lrf.
    # LambdaLR rather than CosineAnnealingLR because LambdaLR recomputes the
    # LR from scratch every epoch, so the per-iteration warmup below can
    # overwrite it without corrupting the schedule.
    lf = lambda e: ((1 - math.cos(e * math.pi / args.epochs)) / 2) * (args.lrf - 1) + 1
    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if args.ema else None
    # YOLOv5 warms up for max(3 epochs, 100 iterations), ramping the LR
    # linearly from 0. Adam's first steps on a from-scratch network are the
    # largest it will ever take, and a bad first few hundred steps can push
    # BatchNorm statistics somewhere the rest of training has to undo.
    n_warmup = max(round(args.warmup_epochs * len(train_loader)), 100) if args.warmup_epochs > 0 else 0
    
    # Creating location to store the outputs
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_seg_loss = float("inf")
    best_det_loss = float("inf")
    history = {"train_total": [], "val_total": [], "cls_acc": []}
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # >>> [YOLOv5-NEW]
        def warmup(i, epoch=epoch):
            ni = i + (epoch - 1) * len(train_loader)
            if ni < n_warmup:
                for g in optimizer.param_groups:
                    g["lr"] = g["initial_lr"] * lf(epoch - 1) * ni / n_warmup

        train_m = train_one_epoch(model, train_loader, optimizer, segmentation_criterion,
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det,
                                  scaler=scaler, use_amp=use_amp,
                                  ema=ema, warmup=warmup if n_warmup else None)     # >>> [YOLOv5-CHANGED]
        # >>> [YOLOv5-CHANGED] validate + checkpoint the EMA weights
        eval_model = ema.ema if ema is not None else model
        val_m = validate(eval_model, validation_loader, segmentation_criterion,
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det,
                                  use_amp=use_amp)
        scheduler.step()

        # so the curve survives even if Colab disconnects mid-training
        history["train_total"].append(train_m["total"])
        history["val_total"].append(val_m["total"])
        history["cls_acc"].append(val_m["cls_acc"])
        with open(output_directory / "history.json", "w") as f:
            json.dump(history, f)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train {train_m['total']:.4f} (train_acc {train_m['cls_acc']:.3f})  "
              f"(det {train_m['det']:.3f} | seg {train_m['seg']:.3f} | cls {train_m['cls']:.3f})  "
              f"val {val_m['total']:.4f}  val_acc {val_m['cls_acc']:.3f}")

        # Multi-task checkpoint rule (lexicographic with tolerance):
        #   primary   — val cls_acc must beat the best so far
        #   guardrail — val seg/det losses may fluctuate, but not more than
        #               (1 + best_tol) of the best they have EVER been
        # This stops us saving an epoch where classification improved but
        # segmentation/detection quietly collapsed.
        best_seg_loss = min(best_seg_loss, val_m["seg"])
        best_det_loss = min(best_det_loss, val_m["det"])
        seg_ok = val_m["seg"] <= best_seg_loss * (1 + args.best_tol) + 0.01
        det_ok = val_m["det"] <= best_det_loss * (1 + args.best_tol) + 0.05

        if val_m["cls_acc"] > best_val_acc and seg_ok and det_ok:
            best_val_acc = val_m["cls_acc"]
            th.save({
                "epoch": epoch,
                "model_state": eval_model.state_dict(),     # >>> [YOLOv5-CHANGED] EMA weights
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_m["total"],
                "val_cls_acc": best_val_acc,
                "val_seg_loss": val_m["seg"],
                "val_det_loss": val_m["det"],
                "args": vars(args),
                "arch": ARCH,     # >>> [YOLOv3-NEW]
                "anchors": model.anchors_list,     # >>> [YOLOv2-NEW] weights are
                # meaningless without the anchors they were decoded against, so
                # they travel together in the same file
            }, output_directory / "best.pth")
            print(f"  ↳ new best (cls_acc {best_val_acc:.3f}, "
                  f"seg {val_m['seg']:.3f}, det {val_m['det']:.3f}), checkpoint saved")
        elif val_m["cls_acc"] > best_val_acc:
            # cls improved but a guardrail failed — say so, don't save silently
            print(f"  ↳ cls_acc improved to {val_m['cls_acc']:.3f} but "
                  f"{'seg' if not seg_ok else 'det'} regressed beyond tolerance, NOT saved")

    # Save the best model/value overall
    th.save({"epoch": args.epochs, "model_state": eval_model.state_dict(),     # >>> [YOLOv5-CHANGED]
             "arch": ARCH,     # >>> [YOLOv3-NEW]
             "anchors": model.anchors_list},     # >>> [YOLOv2-NEW]
               output_directory / "last.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()