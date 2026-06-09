import argparse
from pathlib import Path

import cv2
import numpy as np


def resize_keep_aspect(image, max_side=800):
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def remove_small_components(mask, min_area=500):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 255

    return output


def hsv_initial_mask(
    image,
    s_thresh=(25, 255),
    v_thresh=(30, 245),
    close_size=7,
    min_area=500,
):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    s_mask = cv2.inRange(s_channel, s_thresh[0], s_thresh[1])
    v_mask = cv2.inRange(v_channel, v_thresh[0], v_thresh[1])
    mask = cv2.bitwise_and(s_mask, v_mask)

    kernel = np.ones((close_size, close_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = remove_small_components(mask, min_area=min_area)
    mask = cv2.medianBlur(mask, 5)
    return mask


def grabcut_from_mask(
    image,
    init_mask,
    iter_count=5,
    border_bg_ratio=0.02,
    fallback_rect_margin=0.03,
):
    h, w = image.shape[:2]
    init_ratio = float(np.count_nonzero(init_mask)) / float(init_mask.size)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if init_ratio < 0.01 or init_ratio > 0.95:
        margin_x = int(w * fallback_rect_margin)
        margin_y = int(h * fallback_rect_margin)
        rect = (
            margin_x,
            margin_y,
            max(1, w - 2 * margin_x),
            max(1, h - 2 * margin_y),
        )
        grabcut_mask = np.zeros((h, w), np.uint8)
        cv2.grabCut(
            image,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            iterCount=iter_count,
            mode=cv2.GC_INIT_WITH_RECT,
        )
    else:
        grabcut_mask = np.where(
            init_mask > 0,
            cv2.GC_PR_FGD,
            cv2.GC_PR_BGD,
        ).astype(np.uint8)

        border = max(1, int(min(h, w) * border_bg_ratio))
        grabcut_mask[:border, :] = cv2.GC_BGD
        grabcut_mask[-border:, :] = cv2.GC_BGD
        grabcut_mask[:, :border] = cv2.GC_BGD
        grabcut_mask[:, -border:] = cv2.GC_BGD

        try:
            cv2.grabCut(
                image,
                grabcut_mask,
                None,
                bgd_model,
                fgd_model,
                iterCount=iter_count,
                mode=cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            margin_x = int(w * fallback_rect_margin)
            margin_y = int(h * fallback_rect_margin)
            rect = (
                margin_x,
                margin_y,
                max(1, w - 2 * margin_x),
                max(1, h - 2 * margin_y),
            )
            grabcut_mask = np.zeros((h, w), np.uint8)
            cv2.grabCut(
                image,
                grabcut_mask,
                rect,
                bgd_model,
                fgd_model,
                iterCount=iter_count,
                mode=cv2.GC_INIT_WITH_RECT,
            )

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return refined


def clean_mask(mask, close_size=7, open_size=5, min_area=500, keep_largest=False):
    close_kernel = np.ones((close_size, close_size), np.uint8)
    open_kernel = np.ones((open_size, open_size), np.uint8)

    cleaned = (mask > 0).astype(np.uint8) * 255
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    cleaned = remove_small_components(cleaned, min_area=min_area)

    if not keep_largest:
        return cleaned

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8),
        8,
    )
    if num_labels <= 1:
        return cleaned

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = np.zeros_like(cleaned)
    largest[labels == largest_label] = 255
    return largest


def apply_mask(image, mask):
    binary_mask = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(image, image, mask=binary_mask)


def foreground_ratio(mask):
    return float(np.count_nonzero(mask > 0)) / float(mask.size)


def hsv_grabcut_pipeline(
    image,
    max_side=800,
    return_steps=False,
    keep_largest=False,
):
    resized = resize_keep_aspect(image, max_side=max_side)
    init = hsv_initial_mask(resized)
    refined = grabcut_from_mask(resized, init)
    final = clean_mask(refined, keep_largest=keep_largest)
    segmented = apply_mask(resized, final)

    if return_steps:
        return init, refined, final, segmented
    return final, segmented


def batch_save(
    enhancement="clahe",
    split="val",
    src_root=None,
    dst_root=None,
    keep_largest=False,
):
    project_root = Path(__file__).resolve().parent.parent
    if src_root is None:
        src_root = project_root / "data" / "enhanced" / enhancement / split
    if dst_root is None:
        dst_root = (
            project_root
            / "results"
            / "segmentation"
            / "hsv_grabcut_cleaning"
            / enhancement
            / split
        )

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for class_folder in sorted(src_root.iterdir()):
        if not class_folder.is_dir():
            continue

        out_class = dst_root / class_folder.name
        out_class.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_folder.iterdir()):
            if not img_path.is_file():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            final, segmented = hsv_grabcut_pipeline(
                image,
                keep_largest=keep_largest,
            )
            cv2.imwrite(str(out_class / f"{img_path.stem}_mask.png"), final)
            cv2.imwrite(
                str(out_class / f"{img_path.stem}_segmented.png"),
                segmented,
            )

    return dst_root


def main():
    parser = argparse.ArgumentParser(
        description="HSV + GrabCut + cleaning segmentation pipeline"
    )
    parser.add_argument("--enhancement", default="clahe")
    parser.add_argument("--split", default="val")
    parser.add_argument("--src-root", default=None)
    parser.add_argument("--dst-root", default=None)
    parser.add_argument("--keep-largest", action="store_true")
    args = parser.parse_args()

    output_dir = batch_save(
        enhancement=args.enhancement,
        split=args.split,
        src_root=args.src_root,
        dst_root=args.dst_root,
        keep_largest=args.keep_largest,
    )
    print(f"Saved segmentation results to: {output_dir}")


if __name__ == "__main__":
    main()
