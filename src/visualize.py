import torch as th
import numpy as np
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # No display needed: always render to files (works on Colab / ssh)
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataloader import get_dataLoaders, HandGestureDataset
from model import HandGestureNet, load_handgesture_net     # >>> [YOLOv2-CHANGED]
from utils import extract_gt_box, extract_best_pred_box
from evaluate import evaluate

IMAGE_SIZE = 448


#---------------------------------------------------
# Drawing helpers
#---------------------------------------------------

def center_box_to_rect(box, size=IMAGE_SIZE):
    """(cx, cy, w, h) normalized -> (x_min, y_min, w, h) in pixels, what
    matplotlib's Rectangle patch expects."""
    cx, cy, w, h = [float(v) * size for v in box]
    return cx - w / 2, cy - h / 2, w, h


def draw_sample(image, det_target, det_pred, seg_pred, true_label, pred_label, out_path):
    """
    One figure, two panels:
      left  = RGB + ground truth box (solid) + predicted box (dashed)
      right = RGB + predicted mask (red overlay) + ground truth mask (green contour)
    """
    # [4, H, W] tensor -> [H, W, 3] numpy, keep only RGB channels for display
    rgb = image[:3].permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ---- Left panel: detection ----
    axes[0].imshow(rgb)
    x, y, w, h = center_box_to_rect(extract_gt_box(det_target))
    axes[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime",
                                    linewidth=2, label="ground truth"))
    x, y, w, h = center_box_to_rect(extract_best_pred_box(det_pred))
    axes[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red",
                                    linewidth=2, linestyle="--", label="prediction"))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Detection  |  T: G{true_label + 1:02d}  P: G{pred_label + 1:02d}",
                      color=("green" if true_label == pred_label else "red"))
    axes[0].axis("off")

    # ---- Right panel: segmentation ----
    pred_mask = (th.sigmoid(seg_pred[0]) > 0.5).cpu().numpy()   # logits -> 0/1
    axes[1].imshow(rgb)
    # Predicted mask as semi-transparent red; masked array hides the zeros
    axes[1].imshow(np.ma.masked_where(pred_mask == 0, pred_mask),
                   cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Segmentation (red = predicted hand)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show_predictions(model, loader, device, out_dir, n_samples=8):
    """Run the model on the first n validation images and save one png each."""
    model.eval()
    saved = 0
    with th.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            det_pred, seg_pred, cls_pred = model(images)
            pred_labels = cls_pred.argmax(dim=1)

            for i in range(images.size(0)):
                out_path = out_dir / f"sample_{saved:02d}.png"
                draw_sample(images[i].cpu(), targets["det"][i].cpu(), det_pred[i].cpu(),
                            seg_pred[i].cpu(), targets["Label"][i].item(),
                            pred_labels[i].item(), out_path)
                saved += 1
                if saved >= n_samples:
                    print(f"Saved {saved} sample visualizations to {out_dir}")
                    return


def plot_confusion_matrix(confusion, out_path, n_classes=10):
    """Heatmap: darker cell = more samples. Diagonal should light up."""
    confusion = confusion.numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = [f"G{c + 1:02d}" for c in range(n_classes)]
    ax.set_xticks(range(n_classes), labels, rotation=45)
    ax.set_yticks(range(n_classes), labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix")

    # Write the count inside each cell; switch to white text on dark cells
    threshold = confusion.max() / 2 if confusion.max() > 0 else 1
    for r in range(n_classes):
        for c in range(n_classes):
            ax.text(c, r, str(confusion[r, c]), ha="center", va="center", fontsize=8,
                    color="white" if confusion[r, c] > threshold else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


def plot_loss_curves(history_path, out_path):
    """
    Loss curves from the json that train.py dumps (see README note).
    Skips quietly if the file does not exist.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"No {history_path} found, skipping loss curves "
              "(add the history-saving snippet to train.py to enable this)")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_total"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["train_total"], label="train total")
    ax.plot(epochs, history["val_total"], label="val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curves to {out_path}")


#---------------------------------------------------
# Main method
#---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize HandGestureNet results (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--weights", type=str, default="weights/best.pth")
    parser.add_argument("--history", type=str, default="weights/history.json")
    parser.add_argument("--out-dir", type=str, default="figures/")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # For an EXTERNAL test set: use every person in --data-root, no splitting
    parser.add_argument("--all-data", action="store_true")
    # Same seed as training, same reason as evaluate.py: keep the split identical
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_data:
        test_ds = HandGestureDataset(args.data_root, person_ids=None, augment=False, use_depth=True)
        test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        print(f"External test set: {len(test_ds)} images, ALL persons used")
    else:
        _, _, test_loader = get_dataLoaders(
            args.data_root,
            batch_size=args.batch_size,
            n_val_persons=args.n_val_persons,
            seed=args.seed,
            num_workers=args.num_workers,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )

    # >>> [YOLOv2-CHANGED] rebuild with the checkpoint's own anchors
    checkpoint = th.load(args.weights, map_location=device)
    model = load_handgesture_net(checkpoint, in_channels=4, n_classes=10, device=device)
    print(f"Loaded {args.weights}")

    # 1. Per-image predictions: boxes + masks + labels
    show_predictions(model, test_loader, device, out_dir, n_samples=args.n_samples)

    # 2. Confusion matrix heatmap (reuses evaluate() so numbers always match the report)
    metrics = evaluate(model, test_loader, device)
    plot_confusion_matrix(metrics["confusion"], out_dir / "confusion_matrix.png")

    # 3. Loss curves, if train.py saved a history file
    plot_loss_curves(args.history, out_dir / "loss_curves.png")


if __name__ == "__main__":
    main()