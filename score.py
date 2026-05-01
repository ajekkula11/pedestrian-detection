#!/usr/bin/env python3
"""
score.py — Evaluation script for CSCI 612 Pedestrian Detection
Computes Precision, Recall, F1 using per-box IoU matching at threshold 0.3
Handles empty label files (frames with no pedestrians) correctly.
"""

import os
import glob

LABELS_DIR     = "/home/acv5/pedestrian-detection/dataset/labels"
PREDS_DIR      = "/home/acv5/pedestrian-detection/dataset/predictions"
IOU_THRESHOLD  = 0.3

MIN_PRECISION  = 0.65
MIN_RECALL     = 0.40
TGT_PRECISION  = 0.70
TGT_RECALL     = 0.50


def parse_yolo(filepath, img_w=1, img_h=1):
    """Parse YOLO format file. Returns list of (x1,y1,x2,y2) in pixel coords."""
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((x1, y1, x2, y2))
    return boxes


def compute_iou(a, b):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def evaluate_frame(gt_boxes, pred_boxes, iou_thresh):
    """
    Greedy IoU matching. Returns (TP, FP, FN) for one frame.
    Empty gt + empty pred  → (0, 0, 0)  correct true negative
    Empty gt + some preds  → (0, FP, 0) false positives
    Some gt + empty pred   → (0, 0, FN) missed detections
    """
    matched_gt = set()
    matched_pred = set()

    # Build all IoU pairs, sort by descending IoU
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou >= iou_thresh:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    for iou, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    TP = len(matched_pred)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes)   - TP
    return TP, FP, FN


def main():
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "frame_*.txt")))

    if not label_files:
        print("ERROR: No label files found in", LABELS_DIR)
        return

    total_TP = total_FP = total_FN = 0
    per_frame = []
    missing_preds = 0

    for lf in label_files:
        frame_name = os.path.basename(lf)          # frame_0001.txt
        pred_file  = os.path.join(PREDS_DIR, frame_name)

        gt_boxes   = parse_yolo(lf)
        pred_boxes = parse_yolo(pred_file)

        if not os.path.exists(pred_file):
            missing_preds += 1
            pred_boxes = []

        TP, FP, FN = evaluate_frame(gt_boxes, pred_boxes, IOU_THRESHOLD)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        per_frame.append((frame_name, len(gt_boxes), len(pred_boxes), TP, FP, FN))

    # ── Aggregate metrics ────────────────────────────────────────────────────
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Per-frame breakdown (only non-trivial frames) ────────────────────────
    print("\n=== Per-Frame Breakdown (frames with GT or predictions) ===")
    print(f"{'Frame':<18} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 45)
    for fname, ngt, npred, tp, fp, fn in per_frame:
        if ngt > 0 or npred > 0:   # skip true-negative frames (both empty)
            print(f"{fname:<18} {ngt:>4} {npred:>5} {tp:>4} {fp:>4} {fn:>4}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Frames evaluated : {len(label_files)}")
    print(f"Missing pred files: {missing_preds}")
    print(f"Total TP : {total_TP}")
    print(f"Total FP : {total_FP}")
    print(f"Total FN : {total_FN}")
    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    # ── Tier assessment ──────────────────────────────────────────────────────
    print("\n=== Tier Assessment ===")
    min_pass = precision >= MIN_PRECISION and recall >= MIN_RECALL
    tgt_pass = precision >= TGT_PRECISION and recall >= TGT_RECALL

    print(f"MIN    (P≥{MIN_PRECISION}, R≥{MIN_RECALL}): {'✓ PASS' if min_pass else '✗ FAIL'}")
    print(f"TARGET (P≥{TGT_PRECISION}, R≥{TGT_RECALL}): {'✓ PASS' if tgt_pass else '✗ FAIL'}")

    if tgt_pass:
        print("\n→ You are at TARGET tier. Phase 3 complete.")
    elif min_pass:
        print("\n→ You are at MIN tier. Improve recall to reach TARGET.")
    else:
        print("\n→ Below MIN. Check confidence threshold and label quality.")


if __name__ == "__main__":
    main()
