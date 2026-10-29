import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataloader import get_dataLoaders
from dataloader import HandGestureDataset
from model import HandGestureNet, ARCH     # >>> [YOLOv3-CHANGED]
from utils import compute_intersection_over_union
from utils import shape_iou, anchor_tensor, ANCHORS     # >>> [YOLOv2-NEW]
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


def yolo_detection_loss(prediction, target, layout, lambda_coord=1.0, lambda_obj=1.0,
                        lambda_noobj=1.0, ignore_thresh=0.5):
    """
    >>> [YOLOv3-CHANGED] rewritten for three grids and logistic objectness.

        Steps:
        1. Read the one ground-truth box per image out of the target ->
        2. Pick the responsible ANCHOR by shape over ALL 3 scales (YOLOv2's
           rule, but now the winning anchor also decides the GRID) ->
        3. Find the flat index of (that anchor, the cell holding the centre) ->
        4. Box terms on that one prediction, objectness on every prediction ->
        5. Weighted sum

    What is YOLOv3 about this, compared with the YOLOv2 version:
    - Scale choice is part of the assignment. A big hand matches a big anchor,
      which lives on the 7x7 grid; a small hand ends up on the 28x28 grid.
    - Objectness is BINARY CROSS-ENTROPY with target 1 / 0 ("logistic
      regression" in the paper), not MSE towards the IoU. MSE on a sigmoid has
      a vanishing gradient exactly when the prediction is confidently wrong.
    - IGNORE threshold: a non-responsible prediction that already overlaps the
      hand with IoU > ignore_thresh is neither pushed up nor down. Without it,
      a perfectly good second box on the neighbouring anchor gets punished as
      "background", which fights the responsible box.
    - Box size is weighted by (2 - w*h), YOLOv3's replacement for YOLOv1's
      square-root trick: small boxes count up to 2x, big boxes ~1x.
    - The lambdas are 1 by default: YOLOv3 dropped YOLOv1's 5 / 0.5, because
      with BCE the background terms shrink on their own once they are right.
    """
    N, M, _ = prediction.shape
    gt = gt_boxes_from_target(target)                     # [N, 5]
    has_box = gt[:, 4] > 0.5
    conf = prediction[..., 4]                             # [N, M]

    # ---- objectness targets: 0 everywhere to start with ----
    obj_target = th.zeros_like(conf)
    obj_weight = th.full_like(conf, lambda_noobj)

    if has_box.sum() == 0:
        # Safety check
        return (obj_weight * _bce(conf, obj_target)).sum() / N

    # Ignore mask: predicted boxes already overlapping the truth well enough.
    with th.no_grad():
        iou_all = compute_intersection_over_union(prediction[..., :4], gt[:, None, :4])
        ignore = (iou_all > ignore_thresh) & has_box[:, None]
    obj_weight = obj_weight.masked_fill(ignore, 0.0)

    # ---- step 2/3: responsible anchor -> grid -> cell -> flat index ----
    anchors = layout["anchors"].to(prediction.dtype)
    idx = th.nonzero(has_box).squeeze(1)
    g = gt[idx]
    best = shape_iou(g[:, 2:4], anchors).argmax(dim=1)
    S_b = layout["grid"][best]
    # Which cell of THAT grid holds the centre? min() guards cx == 1.0.
    col = th.minimum((g[:, 0] * S_b).long(), S_b - 1)
    row = th.minimum((g[:, 1] * S_b).long(), S_b - 1)
    flat = layout["offset"][best] + (row * S_b + col) * layout["count"][best] + layout["local"][best]

    obj_target[idx, flat] = 1.0
    obj_weight[idx, flat] = lambda_obj
    loss_obj = (obj_weight * _bce(conf, obj_target)).sum()

    # ---- step 4: box terms on the responsible prediction only ----
    response = prediction[idx, flat]                      # [n, 5]
    Sf = S_b.to(prediction.dtype)
    scale = 2.0 - g[:, 2] * g[:, 3]                       # (2 - w*h)

    # Centre in CELL units (offset inside the cell, in [0, 1]), like YOLOv2.
    pred_off = th.stack([response[:, 0] * Sf - col, response[:, 1] * Sf - row], dim=1)
    gt_off = th.stack([g[:, 0] * Sf - col, g[:, 1] * Sf - row], dim=1)
    loss_xy = (scale[:, None] * (pred_off - gt_off) ** 2).sum()

    # Size in LOG space relative to the anchor, i.e. on tw / th directly, as
    # YOLOv3 does. log(w / anchor_w) is exactly the (clamped) tw the head used.
    aw = anchors[best]
    pred_twh = th.log(response[:, 2:4].clamp(min=1e-6) / aw)
    gt_twh = th.log(g[:, 2:4].clamp(min=1e-6) / aw)
    loss_wh = (scale[:, None] * (pred_twh - gt_twh) ** 2).sum()

    # Returning Multi Part loss function
    return (lambda_coord * (loss_xy + loss_wh) + loss_obj) / N


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
                    scaler=None, use_amp=False):
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

    for images, targets in loader:
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
    
    #Creating model, optimizer, loss
    model = HandGestureNet(in_channels=4, n_classes=10, anchors=anchors).to(device)
    print(f"Detection head: {len(anchors)} anchors (w, h, normalised) = "
          f"{[(round(a, 3), round(b, 3)) for a, b in anchors]}")
    # >>> [YOLOv3-NEW] show which anchors went to which grid
    for S_i, group in zip(model.detecting_head.grids, model.detecting_head.groups):
        print(f"  {S_i:>2}x{S_i:<2} grid: {[(round(a, 3), round(b, 3)) for a, b in group]}")
    segmentation_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Creating location to store the outputs
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_seg_loss = float("inf")
    best_det_loss = float("inf")
    history = {"train_total": [], "val_total": [], "cls_acc": []}
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, segmentation_criterion,
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det,
                                  scaler=scaler, use_amp=use_amp)
        val_m = validate(model, validation_loader, segmentation_criterion,
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
                "model_state": model.state_dict(),
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
    th.save({"epoch": args.epochs, "model_state": model.state_dict(),
             "arch": ARCH,     # >>> [YOLOv3-NEW]
             "anchors": model.anchors_list},     # >>> [YOLOv2-NEW]
               output_directory / "last.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()