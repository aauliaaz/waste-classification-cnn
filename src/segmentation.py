import cv2
import numpy as np
from pathlib import Path
from skimage import morphology
import os


def _resize_keep_aspect(image, max_side=800):
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def otsu_segmentation(image, denoise=True):
    img = _resize_keep_aspect(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    mask = (mask.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def hsv_segmentation(image, s_thresh=(30, 255), v_thresh=(30, 255)):
    img = _resize_keep_aspect(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    s_mask = cv2.inRange(s_channel, s_thresh[0], s_thresh[1])
    v_mask = cv2.inRange(v_channel, v_thresh[0], v_thresh[1])
    mask = cv2.bitwise_and(s_mask, v_mask)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    mask = (mask.astype(np.uint8) * 255)
    mask = cv2.medianBlur(mask, 5)
    return mask


def grabcut_segmentation(image, iter_count=5, rect_margin=0.02):
    img = _resize_keep_aspect(image)
    h, w = img.shape[:2]
    margin_x = int(w * rect_margin)
    margin_y = int(h * rect_margin)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2 = mask2 * 255
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask2


def kmeans_segmentation(image, k=2, attempts=4):
    img = _resize_keep_aspect(image)
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centers = centers.astype(np.uint8)

    # Choose the cluster with highest mean saturation/value as foreground
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].reshape(-1)
    v = hsv[:, :, 2].reshape(-1)
    best_cluster = None
    best_score = -1
    for cluster_idx in range(k):
        mask_cluster = labels == cluster_idx
        if mask_cluster.sum() == 0:
            continue
        score = float(s[mask_cluster].mean() + v[mask_cluster].mean())
        if score > best_score:
            best_score = score
            best_cluster = cluster_idx

    mask = (labels == best_cluster).astype(np.uint8).reshape(img.shape[:2]) * 255
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=300)
    mask = (mask.astype(np.uint8) * 255)
    mask = cv2.medianBlur(mask, 5)
    return mask


def refine_with_grabcut(image, init_mask, iter_count=5):
    img = _resize_keep_aspect(image)
    mask_gc = np.where(init_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    img_copy = img.copy()

    try:
        cv2.grabCut(img_copy, mask_gc, None, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_MASK)
    except Exception:
        h, w = img.shape[:2]
        rect = (int(w * 0.02), int(h * 0.02), int(w * 0.96), int(h * 0.96))
        mask_gc = np.zeros(img.shape[:2], np.uint8)
        cv2.grabCut(img_copy, mask_gc, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask_gc == cv2.GC_BGD) | (mask_gc == cv2.GC_PR_BGD), 0, 1).astype('uint8') * 255
    return mask2

def largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out

def clean_mask(mask, close_size=7, open_size=5, min_area=500):
    kernel_close = np.ones((close_size, close_size), np.uint8)
    kernel_open = np.ones((open_size, open_size), np.uint8)
    m = mask.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def ensemble_pipeline(img, init_method='hsv'):
    if init_method == 'hsv':
        init = hsv_segmentation(img)
    else:
        init = kmeans_segmentation(img)
    refined = refine_with_grabcut(img, init)
    final = clean_mask(refined, close_size=7, open_size=5, min_area=500)
    return init, refined, final


def batch_ensemble_save(enhancement='clahe', split='val', init_method='hsv', src_root=None, dst_root=None):
    project_root = Path(__file__).resolve().parent.parent
    if src_root is None:
        src_root = project_root / 'data' / 'enhanced' / enhancement / split
    if dst_root is None:
        dst_root = project_root / 'results' / 'segmentation' / 'ensemble' / enhancement / split

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
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            _, _, final = ensemble_pipeline(img, init_method=init_method)
            cv2.imwrite(str(out_class / (img_path.stem + '_mask.png')), final)
    return dst_root


def batch_ensemble_all(enhancements=None, splits=None, init_method='hsv'):
    if enhancements is None:
        enhancements = ['clahe', 'gaussian', 'clahe_gaussian']
    if splits is None:
        splits = ['train', 'val', 'test']

    results = {}
    for enhancement in enhancements:
        for split in splits:
            key = f'{enhancement}/{split}'
            dst = batch_ensemble_save(enhancement, split, init_method=init_method)
            results[key] = str(dst)
    return results


def _binarize_mask(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def iou(mask_pred, mask_gt):
    pred = _binarize_mask(mask_pred)
    gt = _binarize_mask(mask_gt)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(intersection) / float(union)


def dice(mask_pred, mask_gt):
    pred = _binarize_mask(mask_pred)
    gt = _binarize_mask(mask_gt)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * float(intersection) / float(denom)


def evaluate_directory(pred_root, gt_root, image_extensions=None):
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']

    pred_root = Path(pred_root)
    gt_root = Path(gt_root)
    metrics = {
        'iou': [],
        'dice': []
    }

    for class_folder in sorted(pred_root.iterdir()):
        if not class_folder.is_dir():
            continue
        gt_class_folder = gt_root / class_folder.name
        if not gt_class_folder.exists():
            continue

        for pred_path in sorted(class_folder.iterdir()):
            if not pred_path.is_file() or pred_path.suffix.lower() not in image_extensions:
                continue
            gt_path = gt_class_folder / pred_path.name
            if not gt_path.exists():
                continue

            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if pred_mask is None or gt_mask is None:
                continue

            metrics['iou'].append(iou(pred_mask, gt_mask))
            metrics['dice'].append(dice(pred_mask, gt_mask))

    if len(metrics['iou']) == 0:
        return {
            'mean_iou': None,
            'mean_dice': None,
            'count': 0
        }
    return {
        'mean_iou': float(np.mean(metrics['iou'])),
        'mean_dice': float(np.mean(metrics['dice'])),
        'count': len(metrics['iou'])
    }


def batch_process(enhancement='clahe', split='val', method='otsu', src_root=None, dst_root=None):
    project_root = Path(__file__).resolve().parent.parent
    if src_root is None:
        src_root = project_root / 'data' / 'enhanced' / enhancement / split
    if dst_root is None:
        dst_root = project_root / 'results' / 'segmentation' / method / enhancement / split

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    methods = {
        'otsu': otsu_segmentation,
        'hsv': hsv_segmentation,
        'grabcut': grabcut_segmentation,
        'kmeans': kmeans_segmentation
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    seg_func = methods[method]

    for class_folder in sorted(src_root.iterdir()):
        if not class_folder.is_dir():
            continue
        out_class = dst_root / class_folder.name
        out_class.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_folder.iterdir()):
            if not img_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                mask = seg_func(img)
            except Exception:
                mask = otsu_segmentation(img)

            out_path = out_class / (img_path.stem + '_mask.png')
            cv2.imwrite(str(out_path), mask)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch segmentation for enhanced dataset')
    parser.add_argument('--enhancement', default='clahe', help='enhancement folder to use')
    parser.add_argument('--split', default='val', help='data split (train/val/test')
    parser.add_argument('--method', default='otsu', help='segmentation method: otsu|hsv|grabcut|kmeans')
    args = parser.parse_args()

    batch_process(enhancement=args.enhancement, split=args.split, method=args.method)
