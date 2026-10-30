"""
YOLOv2-style dimension clusters (anchor priors) for HandGestureNet.

>>> [YOLOv10-NOTE] NOT USED for training on the yolov10 branch: the YOLOv10
detector is anchor-free, so there are no priors to fit. The script is kept
because its box statistics (width / height / aspect ratio of the training
hands) are still a useful description of the data. The anchors themselves
are used on the yolov2 .. yolov5 branches.

WHY THIS EXISTS
---------------
The current DetectionHead is YOLOv1-style: every cell regresses B free boxes
from nothing.  YOLOv2's headline contribution was to stop doing that and
instead give each cell a small set of PRIOR box shapes ("anchors") measured
from the training data with k-means, then predict only a small correction on
top of a prior.  This script is the "measure the training data" half.

The distance used by k-means is  d = 1 - IoU(box, centroid),  NOT Euclidean
distance on (w, h).  We care about overlap, not about raw pixel differences:
a 10-pixel error on a huge box matters far less than on a tiny one.

WHAT IT DOES
------------
1. Rebuilds the SAME person-level train/val/test split as dataloader.py
   (so val/test hands never influence the anchors -- that would be leakage).
2. Re-derives every training ground-truth box exactly the way
   HandGestureDataset does: mask -> resize to 448 (NEAREST) -> tightest box
   -> divide by 448.  So the (w, h) here are in the SAME normalised [0, 1]
   space that encode_yolo_target() and the loss use.
3. Runs IoU k-means for k = 1..K_MAX and reports mean best-IoU for each.
4. Prints the anchor list to paste into src/utils.py, and saves the
   "mean best-IoU vs number of anchors" figure (the YOLOv2 paper plot).

RUN IT (CPU only, no GPU needed, writes nothing except the figure):
    cd src
    python kmeans_anchors.py --data-root ../dataset/ --seed 42
"""

import argparse
import json
import pathlib

import numpy as np
import cv2
import PIL.Image as Image


IMAGE_SIZE = 448        # must match HandGestureDataset.IMAGE_SIZE
K_MAX = 12              # how many anchors to try on the sweep  >>> [YOLOv3-CHANGED] was 8, must reach 9


# ---------------------------------------------------------------------------
# 1. Split -- MUST stay identical to dataloader.get_dataLoaders
# ---------------------------------------------------------------------------

def _diagnose(root):
    """
    Turn 'No rgb frames found' into something you can act on.

    A path can fail three different ways and 'not found' does not tell you
    which one you hit, so we check them one at a time.
    """
    root = pathlib.Path(root)
    lines = ["", "=" * 66,
             "No rgb frames found. Working through the possibilities:",
             "=" * 66,
             f"  you passed        : {root}",
             f"  which resolves to : {root.resolve()}",
             f"  cwd is            : {pathlib.Path.cwd()}",
             ""]

    # 1. Does the folder exist at all? Walk up until something does.
    if not root.exists():
        probe = root.resolve()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        lines.append(f"  That folder does NOT exist. The deepest part that does exist is:")
        lines.append(f"      {probe}")
        try:
            kids = sorted(p.name for p in probe.iterdir() if p.is_dir())[:25]
            lines.append(f"  Folders inside it: {kids}")
        except OSError:
            pass
        lines.append("")
        lines.append("  NOTE: relative paths are resolved from the folder you RUN from,")
        lines.append("  not from the folder the script lives in. Running")
        lines.append("      python src/kmeans_anchors.py --data-root ../data/...")
        lines.append("  from the project root is NOT the same as running")
        lines.append("      cd src ; python kmeans_anchors.py --data-root ../data/...")
        lines.append("  Pass an absolute path if you are unsure.")
        lines.append("=" * 66)
        return "\n".join(lines)

    # 2. It exists -- is there an 'rgb' folder anywhere below it?
    found = []
    for p in root.rglob("rgb"):
        if p.is_dir():
            found.append(p)
        if len(found) >= 5:
            break
    if found:
        lines.append("  The folder exists AND contains 'rgb' folders, but none of them")
        lines.append("  hold any .png files. Examples found:")
        for p in found:
            n = len(list(p.glob("*")))
            ext = sorted({q.suffix for q in p.glob("*")})
            lines.append(f"      {p}   ({n} files, extensions {ext})")
        lines.append("  -> the images are probably .jpg, not .png.")
        lines.append("=" * 66)
        return "\n".join(lines)

    # 3. It exists but has no 'rgb' folder -- show what it does have.
    kids = sorted(p.name for p in root.iterdir())[:25]
    lines.append("  The folder exists but has no 'rgb' subfolder anywhere below it.")
    lines.append(f"  What it contains: {kids}")
    lines.append("")
    lines.append("  Expected layout:")
    lines.append("      <data-root>/G01_call/clip01/rgb/<person>__frame_XXX.png")
    lines.append("      <data-root>/G01_call/clip01/depth/...")
    lines.append("      <data-root>/G01_call/clip01/annotation/...")
    lines.append("  --data-root must point at the folder that CONTAINS G01_call/ etc.")
    lines.append("=" * 66)
    return "\n".join(lines)


def split_persons(root, seed=42, val_frac=0.1, test_frac=0.1):
    """
    Copy of the person-level split in dataloader.get_dataLoaders.

    If you ever change the split there, change it here too, otherwise the
    anchors get fitted on people the model is supposed to be tested on.
    Sanity check: the person lists printed below must match the
    "train persons: / val persons: / test persons:" lines train.py prints.
    """
    root = pathlib.Path(root)
    persons = sorted({p.stem.split("__")[0] for p in root.rglob("rgb/*.png")})
    if not persons:
        raise RuntimeError(_diagnose(root))

    rng = np.random.default_rng(seed)
    rng.shuffle(persons)

    n = len(persons)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac))
    n_val = min(n_val, n - n_test - 1)
    test_ids = set(persons[:n_test])
    val_ids = set(persons[n_test:n_test + n_val])
    train_ids = set(persons[n_test + n_val:])

    print(f"train persons: {sorted(train_ids)}")
    print(f"val persons:   {sorted(val_ids)}")
    print(f"test persons:  {sorted(test_ids)}")
    return train_ids, val_ids, test_ids


def _find_by_stem(folder, stem):
    """Same helper as HandGestureDataset._find_by_stem."""
    if not folder.is_dir():
        return None
    for p in sorted(folder.iterdir()):
        if p.stem == stem:
            return p
    return None


# ---------------------------------------------------------------------------
# 2. Collect the ground-truth box shapes
# ---------------------------------------------------------------------------

def collect_wh(root, person_ids, gesture_filter=None):
    """
    Return an (N, 2) array of (w, h), normalised to [0, 1].

    Mirrors HandGestureDataset.__getitem__ with augment=False:
        mask -> resize 448x448 NEAREST -> tightest box -> / 448
    Augmentation is deliberately OFF: anchors describe the DATA, not the
    random jitter we add on top of it.
    """
    root = pathlib.Path(root)
    wh = []
    n_no_depth = 0
    n_no_mask = 0
    n_empty = 0

    for rgb_path in sorted(root.rglob("rgb/*.png")):
        clip_directory = rgb_path.parent.parent
        gesture_directory = clip_directory.parent
        stem = rgb_path.stem
        person_id = stem.split("__")[0]

        if person_ids is not None and person_id not in person_ids:
            continue

        # dataloader.py drops any frame with no matching depth file, so a box
        # from such a frame would never reach the loss. Drop it here too.
        if _find_by_stem(clip_directory / "depth", stem) is None:
            n_no_depth += 1
            continue

        mask_path = _find_by_stem(clip_directory / "annotation", stem)
        if mask_path is None:
            n_no_mask += 1          # classification-only frame, has no box
            continue

        if gesture_filter and gesture_directory.name not in gesture_filter:
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        rows, cols = np.where(mask == 1)
        if len(rows) == 0:
            n_empty += 1            # blank mask -> dataloader treats as no box
            continue

        w = (cols.max() - cols.min()) / IMAGE_SIZE
        h = (rows.max() - rows.min()) / IMAGE_SIZE
        if w <= 0 or h <= 0:
            n_empty += 1
            continue
        wh.append([w, h])

    print(f"\nboxes collected      : {len(wh)}")
    print(f"skipped, no depth    : {n_no_depth}")
    print(f"skipped, no mask     : {n_no_mask}  (classification-only frames)")
    print(f"skipped, blank mask  : {n_empty}")
    if not wh:
        raise RuntimeError("No boxes collected -- check --data-root")
    return np.asarray(wh, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. IoU k-means
# ---------------------------------------------------------------------------

def iou_wh(boxes, clusters):
    """
    IoU between (N,2) boxes and (K,2) clusters, pretending they share a centre.

    Shape-only IoU: we are asking "is this prior the right SHAPE for this
    hand", not "is it in the right place". Position is handled by the grid.
    """
    inter = (np.minimum(boxes[:, None, 0], clusters[None, :, 0]) *
             np.minimum(boxes[:, None, 1], clusters[None, :, 1]))
    area_b = (boxes[:, 0] * boxes[:, 1])[:, None]
    area_c = (clusters[:, 0] * clusters[:, 1])[None, :]
    return inter / (area_b + area_c - inter + 1e-12)


def kmeans_iou(boxes, k, seed=0, max_iter=300):
    """
    k-means where distance = 1 - IoU instead of Euclidean.

    Centroids are updated with the MEDIAN, not the mean: the box-size
    distribution has a long tail (a few very large hands close to the camera)
    and the median does not get dragged by it.
    """
    rng = np.random.default_rng(seed)
    clusters = boxes[rng.choice(len(boxes), k, replace=False)].copy()
    last = None
    for _ in range(max_iter):
        assign = np.argmax(iou_wh(boxes, clusters), axis=1)
        if last is not None and (assign == last).all():
            break
        for j in range(k):
            member = boxes[assign == j]
            if len(member):
                clusters[j] = np.median(member, axis=0)
            else:
                # Dead cluster: re-seed it on the worst-covered box so we
                # never silently return fewer than k usable anchors.
                worst = np.argmin(iou_wh(boxes, clusters).max(axis=1))
                clusters[j] = boxes[worst]
        last = assign
    mean_best_iou = iou_wh(boxes, clusters).max(axis=1).mean()
    return clusters, mean_best_iou


def best_of_n_restarts(boxes, k, restarts=15):
    """k-means is seed-sensitive, so keep the best of several restarts."""
    best_clusters, best_score = None, -1.0
    for s in range(restarts):
        clusters, score = kmeans_iou(boxes, k, seed=s)
        if score > best_score:
            best_clusters, best_score = clusters, score
    order = np.argsort(best_clusters[:, 0] * best_clusters[:, 1])   # small -> large
    return best_clusters[order], best_score


# ---------------------------------------------------------------------------
# 4. Baselines to compare against
# ---------------------------------------------------------------------------

def handpicked_baseline(boxes, k=2):
    """
    The honest 'what would I have guessed' baseline: k SQUARE anchors placed
    at evenly spaced percentiles of sqrt(area). This is the anchor set a
    reasonable person picks without looking at the data properly.
    """
    side = np.sqrt(boxes[:, 0] * boxes[:, 1])
    qs = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    sides = np.quantile(side, qs)
    return np.stack([sides, sides], axis=1)


def aspect_aware_baseline(boxes, k):
    """
    The HONEST baseline: what someone gets after looking at the data for
    thirty seconds, without running k-means at all.

    They would notice that hands are taller than they are wide, take the
    median h/w ratio, and spread k boxes across the observed size range.
    Comparing k-means against SQUARE anchors is a straw man -- almost all of
    that gain is just "hands are not square", which needs no clustering.
    The number worth reporting is k-means MINUS this.
    """
    side = np.sqrt(boxes[:, 0] * boxes[:, 1])
    r = np.median(boxes[:, 1] / boxes[:, 0])          # h / w
    qs = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    sides = np.quantile(side, qs)
    return np.stack([sides / np.sqrt(r), sides * np.sqrt(r)], axis=1)


def oob_score(boxes, k, n_boot=15, seed=0, restarts=4):
    """
    Out-of-bag estimate of mean best-IoU -- an HONEST version of the number.

    Why we need it: the printed mean best-IoU is measured on the very boxes
    the anchors were fitted to, so it can only ever look good. With 699 boxes
    and k=5 that is roughly 140 boxes per anchor, which is not a lot.

    The fix is the standard bootstrap trick:
        draw 699 boxes WITH replacement  -> fit anchors on those
        the ~37% never drawn ("out of bag") -> score the anchors on those
    The out-of-bag boxes had no say in where the anchors went, so the score
    is an estimate of how these anchors handle a hand they have not seen.

    Returns (mean, standard deviation) over n_boot repeats.
    """
    rng = np.random.default_rng(seed)
    n = len(boxes)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 10:
            continue
        clusters, _ = best_of_n_restarts(boxes[idx], k, restarts=restarts)
        scores.append(iou_wh(boxes[oob], clusters).max(axis=1).mean())
    return float(np.mean(scores)), float(np.std(scores))


def anchor_stability(boxes, k, n_boot=15, seed=1, restarts=4):
    """
    How much do the anchors themselves move if the data had been slightly
    different? Refit on bootstrap resamples and report, per anchor, the
    average relative wobble in w and h.

    An anchor that moves by 30% between resamples is not a real cluster in
    the data, it is an artefact of these particular 699 boxes.
    """
    base, _ = best_of_n_restarts(boxes, k)
    rng = np.random.default_rng(seed)
    n = len(boxes)
    devs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        clusters, _ = best_of_n_restarts(boxes[idx], k, restarts=restarts)
        # both are sorted small -> large by area, so index i matches index i
        devs.append(np.abs(clusters - base) / base)
    return base, np.mean(devs, axis=0)


def report(boxes, clusters, label):
    score = iou_wh(boxes, clusters).max(axis=1).mean()
    print(f"  {label:<34} mean best-IoU = {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv2 dimension clusters for HandGestureNet")
    parser.add_argument("--data-root", type=str, default="../dataset/")
    # These three MUST match what you pass to train.py, or the anchors are
    # fitted on people the model will later be tested on.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # >>> [YOLOv3-CHANGED] default 5 -> 9. YOLOv3 uses 9 clusters, 3 per
    # prediction grid; model.py hands the 3 biggest to 7x7, the next 3 to
    # 14x14 and the 3 smallest to 28x28 (utils.split_anchors_by_scale).
    parser.add_argument("--k", type=int, default=9,
                        help="number of anchors to emit (YOLOv2 used 5, YOLOv3 uses 9)")
    # Relative to the folder you RUN from. Default assumes the project root,
    # which is where you have actually been running it from.
    parser.add_argument("--fig-out", type=str, default="figures/anchor_kmeans.png")
    # >>> [YOLOv2-NEW] writing the anchors to a file means train.py can read them
    # directly. On Kaggle/Colab you cannot conveniently hand-edit utils.py inside
    # a freshly cloned repo, and a stale hand-pasted list is a silent bug.
    parser.add_argument("--out-json", type=str, default="",
                        help="also write the anchors (plus provenance) to this JSON")
    args = parser.parse_args()

    train_ids, _, _ = split_persons(args.data_root, args.seed,
                                    args.val_frac, args.test_frac)
    boxes = collect_wh(args.data_root, train_ids)

    # --- what does the data actually look like? ----------------------------
    w, h = boxes[:, 0], boxes[:, 1]
    ratio = h / w
    print("\nBox statistics (normalised to image size, TRAIN split only)")
    print(f"  width   median {np.median(w):.3f}   "
          f"5-95% [{np.quantile(w, 0.05):.3f}, {np.quantile(w, 0.95):.3f}]")
    print(f"  height  median {np.median(h):.3f}   "
          f"5-95% [{np.quantile(h, 0.05):.3f}, {np.quantile(h, 0.95):.3f}]")
    print(f"  h/w     median {np.median(ratio):.3f}   "
          f"5-95% [{np.quantile(ratio, 0.05):.3f}, {np.quantile(ratio, 0.95):.3f}]")

    # --- how much does each extra anchor buy? ------------------------------
    # In-sample mean best-IoU can only go UP as k grows, so on its own it is
    # useless for choosing k. The out-of-bag column is the one to read: it is
    # scored on boxes that had no say in where the anchors went.
    print("\nmean best-IoU vs number of anchors (the YOLOv2 elbow plot)")
    print(f"  {'k':>3}  {'in-sample':>10}  {'gain':>8}  {'out-of-bag':>18}  {'gap':>7}")
    ks, scores, oob_means, oob_sds, all_clusters = [], [], [], [], {}
    for k in range(1, K_MAX + 1):
        clusters, score = best_of_n_restarts(boxes, k)
        om, osd = oob_score(boxes, k)
        ks.append(k)
        scores.append(score)
        oob_means.append(om)
        oob_sds.append(osd)
        all_clusters[k] = clusters
        gain = "     -  " if k == 1 else f"{score - scores[-2]:+8.4f}"
        print(f"  {k:>3}  {score:>10.4f}  {gain}  "
              f"{om:>10.4f} +/- {osd:.4f}  {score - om:>7.4f}")
    print("  'gap' = how much the in-sample number flatters itself. A gap that")
    print("  grows with k means the extra anchors are memorising these boxes.")

    # --- the decision ------------------------------------------------------
    k = args.k
    clusters = all_clusters[k]
    print(f"\nAnchor quality at k = {k} (higher is better):")
    report(boxes, handpicked_baseline(boxes, 2), "naive: 2 squares")
    basek = report(boxes, handpicked_baseline(boxes, k), f"naive: {k} squares")
    aspect = report(boxes, aspect_aware_baseline(boxes, k),
                    f"aspect-aware: {k} boxes  <- REAL baseline")
    kmean = report(boxes, clusters, f"k-means: {k} anchors")
    om, osd = oob_score(boxes, k)
    print(f"  {'k-means, out-of-bag  <- REAL number':<34} mean best-IoU = "
          f"{om:.4f} +/- {osd:.4f}")

    print(f"\n  vs naive squares      : {kmean - basek:+.4f}   <- straw man, do NOT report this")
    print(f"  vs aspect-aware       : {kmean - aspect:+.4f}   <- what k-means actually bought you")
    print(f"  in-sample optimism    : {kmean - om:+.4f}   <- how much the headline flatters itself")
    print("\n  Almost all of the gain over SQUARE anchors is just 'hands are")
    print("  taller than they are wide', which you can see by eye. The line")
    print("  that belongs in the report is the aspect-aware one.")
    print("  Rule of thumb: below +0.02 there the priors are not buying much,")
    print("  and the honest thing is to report that instead of retraining.")

    # --- are these anchors real, or an artefact of 699 boxes? --------------
    base, wobble = anchor_stability(boxes, k)
    print(f"\nAnchor stability under bootstrap resampling ({len(boxes)} boxes):")
    print(f"  {'anchor (w, h)':<22} {'wobble w':>9} {'wobble h':>9}")
    for (aw, ah), (dw, dh) in zip(base, wobble):
        flag = "   <- unstable" if max(dw, dh) > 0.15 else ""
        print(f"  ({aw:.4f}, {ah:.4f})      {dw:>8.1%}  {dh:>8.1%}{flag}")
    print("  Wobble = average relative move when the data is resampled.")
    print("  Above ~15% the 'cluster' is mostly these particular boxes talking.")

    print("\nPaste into src/utils.py:\n")
    print("ANCHORS = [")
    for cw, ch in clusters:
        print(f"    ({cw:.4f}, {ch:.4f}),")
    print("]")

    # >>> [YOLOv2-NEW] machine-readable copy, so train.py --anchors can consume
    # it and nobody has to retype five pairs of numbers correctly.
    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "anchors": [[round(float(cw), 4), round(float(ch), 4)] for cw, ch in clusters],
            "k": k,
            "n_boxes": int(len(boxes)),
            "mean_best_iou_in_sample": round(float(kmean), 4),
            "mean_best_iou_oob": round(float(om), 4),
            "mean_best_iou_oob_sd": round(float(osd), 4),
            "aspect_aware_baseline": round(float(aspect), 4),
            "gain_over_aspect_aware": round(float(kmean - aspect), 4),
            "split": {"seed": args.seed, "val_frac": args.val_frac,
                      "test_frac": args.test_frac},
            "train_persons": sorted(train_ids),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nAnchors + provenance written to {out}")

    # --- the figure --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = pathlib.Path(args.fig_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

        om_arr = np.asarray(oob_means)
        sd_arr = np.asarray(oob_sds)
        ax[0].plot(ks, scores, "o-", label="in-sample (optimistic)")
        ax[0].plot(ks, om_arr, "s-", label="out-of-bag (honest)")
        ax[0].fill_between(ks, om_arr - sd_arr, om_arr + sd_arr, alpha=0.2)
        ax[0].axhline(iou_wh(boxes, aspect_aware_baseline(boxes, k)).max(axis=1).mean(),
                      ls=":", c="k", label=f"aspect-aware baseline (k={k})")
        ax[0].axvline(k, ls="--", c="grey")
        ax[0].set_xlabel("number of anchors (k)")
        ax[0].set_ylabel("mean best-IoU")
        ax[0].set_title("How much does each extra anchor buy?")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        ax[1].scatter(w, h, s=4, alpha=0.15, label="GT boxes")
        ax[1].scatter(clusters[:, 0], clusters[:, 1], s=90, marker="X",
                      c="red", label=f"k-means anchors (k={k})")
        ax[1].set_xlabel("width (normalised)")
        ax[1].set_ylabel("height (normalised)")
        ax[1].set_title("Where the anchors land in the data")
        ax[1].legend()
        ax[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"\nFigure saved to {out}")
    except ImportError:
        print("\n(matplotlib not installed -- skipped the figure)")


if __name__ == "__main__":
    main()
