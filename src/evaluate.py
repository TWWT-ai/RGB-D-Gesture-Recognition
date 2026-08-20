import torch as th
import argparse
from pathlib import Path
 
from torch.utils.data import DataLoader
from dataloader import get_dataLoaders, HandGestureDataset
from model import HandGestureNet, load_handgesture_net     # >>> [YOLOv2-CHANGED]
from utils import compute_intersection_over_union, extract_best_pred_box, extract_gt_box


# Grid size, must match training
S = 7

#---------------------------------------------------
# Per-head metric helpers
#---------------------------------------------------

def segmentation_scores(seg_logits, gt_mask, eps=1e-6):
    """
    Pixel IoU and Dice for one batch.
    seg_logits: [N, 1, H, W] raw logits (BCEWithLogitsLoss means the model
    outputs logits, so we must apply sigmoid HERE before thresholding).
    """
    pred_mask = (th.sigmoid(seg_logits) > 0.5).float()
    gt_mask = gt_mask.float()
 
    # Flatten each image, count overlaps per image, then sum over batch
    inter = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    pred_area = pred_mask.sum(dim=(1, 2, 3))
    gt_area = gt_mask.sum(dim=(1, 2, 3))
    
    iou = (inter / (pred_area + gt_area - inter + eps)).sum().item()
    dice_coef = (2 * inter / (pred_area + gt_area + eps)).sum().item()
    return iou, dice_coef

#---------------------------------------------------
# Evaluation loop
#---------------------------------------------------
 
@th.no_grad()
def evaluate(model, loader, device, n_classes=10, iou_thresh=0.5):
    model.eval()
 
    # Accumulators
    confusion = th.zeros(n_classes, n_classes, dtype=th.long)   # rows = truth, cols = prediction
    sum_seg_iou, sum_seg_dice = 0.0, 0.0
    sum_det_iou, det_hits = 0.0, 0
    total = 0
 
    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        det_pred, seg_pred, cls_pred = model(images)
        n = images.size(0)
        total += n
 
        # ---- Classification: fill the confusion matrix ----
        pred_label = cls_pred.argmax(dim=1)
        for t, p in zip(targets["Label"], pred_label):
            confusion[t.item(), p.item()] += 1
 
        # ---- Segmentation: pixel IoU + Dice ----
        batch_iou, batch_dice = segmentation_scores(seg_pred, targets["Mask"])
        sum_seg_iou += batch_iou
        sum_seg_dice += batch_dice
 
        # ---- Detection: best box vs ground truth box, one pair per image ----
        for i in range(n):
            gt_box = extract_gt_box(targets["det"][i])
            pred_box = extract_best_pred_box(det_pred[i])
            iou = compute_intersection_over_union(pred_box, gt_box).item()
            sum_det_iou += iou
            if iou >= iou_thresh:
                det_hits += 1
 
    # ---- Aggregate ----
    per_class_correct = confusion.diag().float()
    per_class_total = confusion.sum(dim=1).float()
 
    metrics = {
        "cls_acc": (per_class_correct.sum() / total).item(),
        "cls_acc_per_class": (per_class_correct / per_class_total.clamp(min=1)).tolist(),
        "confusion": confusion,
        "seg_iou": sum_seg_iou / total,
        "seg_dice": sum_seg_dice / total,
        "det_mean_iou": sum_det_iou / total,
        "det_acc": det_hits / total,          # fraction of images with IoU >= iou_thresh
        "n_samples": total,
    }
    return metrics


def print_report(m, iou_thresh=0.5):
    print("=" * 52)
    print(f"Evaluation on {m['n_samples']} held-out TEST images (unseen person)")
    print("=" * 52)
    print(f"[Classification] accuracy       : {m['cls_acc']:.4f}")
    for c, acc in enumerate(m["cls_acc_per_class"]):
        print(f"                 G{c + 1:02d} accuracy   : {acc:.4f}")
    print(f"[Segmentation]   pixel IoU      : {m['seg_iou']:.4f}")
    print(f"                 Dice           : {m['seg_dice']:.4f}")
    print(f"[Detection]      mean IoU       : {m['det_mean_iou']:.4f}")
    print(f"                 acc @ IoU>={iou_thresh}: {m['det_acc']:.4f}")
    print("\nConfusion matrix (rows = truth G01..G10, cols = prediction):")
    header = "     " + " ".join(f"P{c + 1:02d}" for c in range(m["confusion"].size(1)))
    print(header)
    for r in range(m["confusion"].size(0)):
        row = " ".join(f"{v:3d}" for v in m["confusion"][r].tolist())
        print(f"G{r + 1:02d}  {row}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HandGestureNet (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--weights", type=str, default="weights/best.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    # For an EXTERNAL test set: use every person in --data-root, no splitting
    parser.add_argument("--all-data", action="store_true")
    # MUST match the seed used in training, otherwise the person split
    # changes and "validation" persons leak from the training set
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
 
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    if args.all_data:
        # External test set: EVERYTHING in data-root is test data, no split.
        # No augmentation, no shuffle — same rules as any evaluation.
        test_ds = HandGestureDataset(args.data_root, person_ids=None, augment=False, use_depth=True)
        test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        print(f"External test set: {len(test_ds)} images, ALL persons used")
    else:
        # Internal split: same seed as training -> same held-out test person
        _, _, test_loader = get_dataLoaders(
            args.data_root,
            batch_size=args.batch_size,
            n_val_persons=args.n_val_persons,
            seed=args.seed,
            num_workers=args.num_workers,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
 
    # Rebuild the empty model, then pour the trained weights back in
    # >>> [YOLOv2-CHANGED] the anchor set now comes out of the checkpoint instead
    # of a hard-coded B=2, so evaluation can never use different priors from training
    checkpoint = th.load(args.weights, map_location=device)
    model = load_handgesture_net(checkpoint, in_channels=4, n_classes=10, device=device)
    print(f"Loaded {args.weights} (epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss {checkpoint.get('val_loss', float('nan')):.4f})")
    print(f"Anchors: {[(round(a, 3), round(b, 3)) for a, b in model.anchors_list]}")   # >>> [YOLOv2-NEW]
 
    metrics = evaluate(model, test_loader, device, iou_thresh=args.iou_thresh)
    print_report(metrics, iou_thresh=args.iou_thresh)
 
 
if __name__ == "__main__":
    main()