import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataloader import get_dataLoaders
from dataloader import HandGestureDataset
from model import HandGestureNet
from utils import compute_intersection_over_union
from utils import shape_iou, anchor_tensor, ANCHORS     # >>> [YOLOv2-NEW]
# Grid size
S = 7

#---------------------------------------------------
# Loss Function
#---------------------------------------------------

def yolo_detection_loss(prediction, target, B=None, lambda_coord=5.0, lambda_noobj=0.5,
                        anchors=None):
    """
        Steps:
        1. Split cells into two groups ->
        2. Push confidence toward zero for no-object cells ->
        3. Responsible-box selection ->
        4. Computing the three loss term: xy, wh, conf ->
        5. Weighted sum

    >>> [YOLOv2-CHANGED] Step 3 is the part that changed.
    YOLOv1 picked the responsible box by asking which of the B PREDICTIONS
    currently overlaps the ground truth best. That is circular: at epoch 1 the
    predictions are noise, so the choice is noise, and two boxes can keep
    swapping roles for many epochs.
    YOLOv2 picks it by asking which ANCHOR has the closest SHAPE to the
    ground-truth box. The anchors never change during training, so a
    tall-thin hand is always handled by the tall-thin anchor. Each anchor
    gets to specialise instead of all of them chasing the average box.
    """

    N = prediction.size(0)
    # >>> [YOLOv2-CHANGED] B is the anchor count now; read it off the tensor
    # so it can never disagree with the head.
    if B is None:
        B = prediction.shape[-1] // 5
    if anchors is None:
        anchors = anchor_tensor(ANCHORS, device=prediction.device, dtype=prediction.dtype)
    anchors = anchors.to(device=prediction.device, dtype=prediction.dtype)
    # Unzip the combined last index back to Box (B) and (x, y, w, h, confidence)
    prediction = prediction.view(N, S, S, B, 5)

    # Detect whether there is an object within the boxes
    object_mask = target[..., 4] > 0.5
    no_object_mask = ~object_mask             # Flipping the mask

    # Selecting total numbers of all the boxes with the object and the confidence of the object
    no_object_conf = prediction[no_object_mask][..., 4]
    # Creating a no_object_conf size like (assumed correct answer) but filled with 0, containing all cells without object
    # Anything not equal to 0 will be considered loss/wrong prediction, to reduce the falseness reflected on to the last model
    loss_no_object = F.mse_loss(no_object_conf, th.zeros_like(no_object_conf), reduction="sum")

    if object_mask.sum() == 0:
        # Safety check
        return lambda_noobj * loss_no_object / N

    # If there is something within the box
    prediction_object = prediction[object_mask]
    target_object = target[object_mask]

    # >>> [YOLOv2-CHANGED] responsible box = best SHAPE match to an anchor.
    # Was: argmax over IoU of the current predictions (see docstring).
    best_idx = shape_iou(target_object[:, 2:4], anchors).argmax(dim=1)

    # Fancy indexing: using the selected best
    response = prediction_object[th.arange(prediction_object.size(0)), best_idx]

    # >>> [YOLOv2-NEW] the confidence target is still "how well does the box
    # you actually predicted overlap the truth", so we still need that IoU --
    # we just no longer use it to CHOOSE the box.
    best_iou = compute_intersection_over_union(response[:, :4], target_object[:, :4])

    # Calculating loss 
    loss_xy = F.mse_loss(response[:, 0:2], target_object[:, 0:2], reduction="sum")
    # Square root would punish small boundary boxes more 
    loss_wh = F.mse_loss(
        th.sqrt(response[:, 2:4].clamp(min=1e-6)),
        th.sqrt(target_object[:, 2:4].clamp(min=1e-6)),
        reduction="sum",
    )
    loss_confidence = F.mse_loss(response[:, 4], best_iou.detach(), reduction="sum")

    # Returning Multi Part loss function
    return (lambda_coord * (loss_xy + loss_wh) + loss_confidence + lambda_noobj * loss_no_object) / N


def compute_loss(outputs, targets, segmentation_criterion, classification_criterion, lambda_seg=1.0, lambda_cls=1.0, lambda_det=1.0,
                 anchors=None):     # >>> [YOLOv2-NEW]
    # Computing loss function for each Head defined in models
    detection_pred, segmentation_pred, classification_pred = outputs

    # Classification is trained on EVERY sample (annotated or not) -> uses all data
    loss_classification = classification_criterion(classification_pred, targets["Label"])

    # Detection + segmentation only make sense where we actually have a mask/box.
    # has_mask is a per-sample flag (1.0 = annotated, 0.0 = classification-only).
    has_mask = targets["has_mask"].bool()
    if has_mask.any():
        loss_detection = yolo_detection_loss(detection_pred[has_mask], targets["det"][has_mask],
                                             anchors=anchors)     # >>> [YOLOv2-NEW]
        loss_segmentation = segmentation_criterion(segmentation_pred[has_mask],
                                                   targets["Mask"][has_mask].float())
    else:
        # No annotated sample in this batch -> nothing to learn for det/seg here
        zero = classification_pred.new_zeros(())
        loss_detection = zero
        loss_segmentation = zero

    total = lambda_det * loss_detection + lambda_seg * loss_segmentation + lambda_cls * loss_classification
    losses = {"det": float(loss_detection),
              "seg": float(loss_segmentation),
              "cls": float(loss_classification),
              "total": float(total)
              }
    return total, losses


#---------------------------------------------------
# Train and validation
#---------------------------------------------------

def train_one_epoch(model, loader, optimizer, segmentation_criterion,
                    classification_criterion, device, lambda_segmentation,
                    lambda_classification, lambda_detection=1.0):
    model.train()
    # >>> [YOLOv2-NEW] read the anchors off the model, so the loss can never
    # be matched against a different anchor set from the one the head decoded with
    anchors = model.detecting_head.anchors
    running = {"det": 0.0,
              "seg": 0.0,
              "cls": 0.0,
              "total": 0.0
              }
    correct, total_samples = 0, 0          # track TRAIN classification accuracy

    for images, targets in loader:
        # Copying what is on CPU onto GPU (RAM to VRAM)
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # Computing loss for each image
        outputs = model(images)
        loss, parts = compute_loss(outputs, targets, segmentation_criterion,
                                   classification_criterion, lambda_segmentation,
                                   lambda_classification, lambda_detection,
                                   anchors=anchors)     # >>> [YOLOv2-NEW]

        # Cleaning the previously loaded data first
        optimizer.zero_grad()
        # The chain rule part, this tells you on which way you should adjust the direction
        # dloss/dw = dloss/dy * dy/dz * dz/dw
        loss.backward()
        # Using Adam formula to update the and override the old data
        optimizer.step()

        # Adding the loss to the training model
        for k in running:
            running[k] += parts[k]

        # Train classification accuracy (compare against val to spot over/underfitting)
        pred_label = outputs[2].argmax(dim=1)
        correct += (pred_label == targets["Label"]).sum().item()
        total_samples += images.size(0)

    n = len(loader)
    metrics = {k: v / n for k, v in running.items()}
    metrics["cls_acc"] = correct / total_samples
    return metrics

@th.no_grad()
def validate(model, loader, segmentation_criterion,
            classification_criterion, device, lambda_segmentation,
            lambda_classification, lambda_detection=1.0):
    model.eval()
    anchors = model.detecting_head.anchors     # >>> [YOLOv2-NEW]
    running = {"det": 0.0,
              "seg": 0.0,
              "cls": 0.0, 
              "total": 0.0
              }
    correct, total_samples = 0, 0

    for images, targets in loader:
        # Copying what is on CPU onto GPU (RAM to VRAM)
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(images)
        _, parts = compute_loss(outputs, targets, segmentation_criterion,
                                   classification_criterion, lambda_segmentation,
                                   lambda_classification, lambda_detection,
                                   anchors=anchors)     # >>> [YOLOv2-NEW]

        # Adding the loss to the training model
        for k in running:
            running[k] += parts[k]

        # Classification accuracy 
        pred_label = outputs[2].argmax(dim=1)
        correct += (pred_label == targets["Label"]).sum().item()
        total_samples += images.size(0)

    n = len(loader)
    metrics = {k: v / n for k, v in running.items()}
    metrics["cls_acc"] = correct / total_samples
    return metrics


#---------------------------------------------------
# Main method
#---------------------------------------------------

def main():
    # Like creating an empty sheet with all the data we are tracking
    parser = argparse.ArgumentParser(description="Train HandGestureNet (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-det", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # Checkpoint rule: cls_acc must improve AND seg/det val losses must stay
    # within (1 + best-tol) of the best they have ever been
    parser.add_argument("--best-tol", type=float, default=0.15)
    parser.add_argument("--out-dir", type=str, default="weights/")
    parser.add_argument("--seed", type=int, default=42)
    # Pack all the top argument into one
    args = parser.parse_args()

    # Keeping the training sample set the same every training at the start
    th.manual_seed(args.seed)
    # Selecting the available GPU device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    )
    
    #Creating model, optimizer, loss
    # >>> [YOLOv2-CHANGED] B=2 free boxes -> len(ANCHORS) anchor-based boxes.
    # Run kmeans_anchors.py first and paste its output into utils.ANCHORS.
    model = HandGestureNet(in_channels=4, n_classes=10, anchors=ANCHORS).to(device)
    print(f"Detection head: {len(ANCHORS)} anchors (w, h, normalised) = "
          f"{[(round(a, 3), round(b, 3)) for a, b in ANCHORS]}")
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
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det)
        val_m = validate(model, validation_loader, segmentation_criterion,
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det)
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
             "anchors": model.anchors_list},     # >>> [YOLOv2-NEW]
               output_directory / "last.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()