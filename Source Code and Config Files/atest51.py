import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

# ================= 1. 全局配置（直接用你的路径，无需修改）=================
PYTORCH_MODEL_PATHS = {
    "YOLO11n-CT (PyTorch)": "/root/root/out/output2/yolo11n_ct_aug_v1/weights/best.pt",
    "YOLO11n-Normal (PyTorch)": "/root/root/output2/yolo11n_aug_v1/weights/best.pt",
}

ONNX_MODEL_PATHS = {
    "YOLO11n-CT (ONNX)": "/root/root/out/output2/yolo11n_ct_aug_v1/weights/best.onnx",
    "YOLO11n-Normal (ONNX)": "/root/root/output2/yolo11n_aug_v1/weights/best.onnx",
}

YAML_PATH = "/root/root/split/data.yaml"
OUTPUT_ROOT = "/root/root/detection_results_unified_onnx"
TEST_IMG_DIR = "/root/root/split/images/test"
TEST_LBL_DIR = "/root/root/split/labels/test"

# ================= 2. 推理配置（所有模型完全一致）=================
IMG_SIZE = 544
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = 0
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

IOU_MATCH_THRES = 0.5
MATCH_CLASS = True
CLEAR_LABEL_CACHE = True
AUTO_EXPORT_ONNX = True
FORCE_ONNX_CPU = False

# 是否同时用 Ultralytics 官方 val 验证 ONNX。默认关闭，避免重复耗时。
RUN_ULTRALYTICS_ONNX_VAL = False


# ================= 3. 环境与ONNX Runtime工具 =================
def print_runtime_info() -> None:
    print("\n🧩 运行环境")
    print(f"  Python PID:     {os.getpid()}")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  PyTorch CUDA:   {torch.version.cuda}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  ONNXRuntime:    {ort.__version__}")
    print(f"  ORT providers:  {ort.get_available_providers()}")
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            print(f"  GPU device:     cuda:{DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
        except Exception as exc:
            print(f"  GPU init warn:  {exc}")


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    """
    创建ONNXRuntime Session，并严格确认CUDAExecutionProvider是否真正启用。
    之前脚本的问题是 CUDA EP 加载失败后仍打印“GPU已启用”，这里改成以 sess.get_providers() 为准。
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

    available = ort.get_available_providers()
    use_cuda = (
        not FORCE_ONNX_CPU
        and torch.cuda.is_available()
        and "CUDAExecutionProvider" in available
    )

    if use_cuda:
        try:
            torch.cuda.init()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": DEVICE}, {}]
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=providers,
                provider_options=provider_options,
            )
            actual = sess.get_providers()
            if "CUDAExecutionProvider" in actual:
                print(f"✅ ONNXRuntime CUDA已启用: {actual}")
                return sess
            print(f"⚠️ ONNXRuntime未实际启用CUDA，当前providers={actual}，切换CPU")
        except Exception as exc:
            print(f"⚠️ ONNXRuntime CUDA初始化失败，切换CPU: {exc}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"✅ ONNXRuntime CPU模式: {sess.get_providers()}")
    return sess


# ================= 4. 基础工具函数 =================
def clear_label_cache() -> None:
    label_dir = Path(TEST_LBL_DIR)
    cache_candidates = {
        label_dir.with_suffix(".cache"),
        label_dir.parent / f"{label_dir.name}.cache",
    }
    for cache_path in cache_candidates:
        if cache_path.exists():
            cache_path.unlink()
            print(f"🧹 清理旧标签缓存: {cache_path}")


def get_test_images() -> list[Path]:
    img_dir = Path(TEST_IMG_DIR)
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def count_test_images() -> int:
    return len(get_test_images())


def clip_box_xyxy(box: tuple[float, float, float, float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(round(max(0, min(float(x1), img_w - 1))))
    y1 = int(round(max(0, min(float(y1), img_h - 1))))
    x2 = int(round(max(0, min(float(x2), img_w - 1))))
    y2 = int(round(max(0, min(float(y2), img_h - 1))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def xywhn_to_xyxy(values: list[float], img_w: int, img_h: int) -> dict:
    cls, x, y, w, h = values
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return {"class": int(cls), "bbox": clip_box_xyxy((x1, y1, x2, y2), img_w, img_h)}


def load_gt_boxes(img_path: Path) -> list[dict]:
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    img_h, img_w = img.shape[:2]
    label_path = Path(TEST_LBL_DIR) / f"{img_path.stem}.txt"
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    gt_boxes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"⚠️ 跳过异常标签行: {label_path}:{line_no} -> {line.strip()}")
                continue
            try:
                values = list(map(float, parts))
                gt_boxes.append(xywhn_to_xyxy(values, img_w, img_h))
            except ValueError:
                print(f"⚠️ 跳过无法解析的标签行: {label_path}:{line_no} -> {line.strip()}")
    return gt_boxes


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """纯NumPy NMS，boxes为xyxy，返回保留索引。"""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        order = rest[iou <= iou_thres]

    return np.asarray(keep, dtype=np.int64)


# ================= 5. PyTorch/ONNX预测框解析 =================
def extract_pytorch_pred_boxes(result, img_shape: tuple[int, int]) -> list[dict]:
    """
    解析Ultralytics PyTorch推理结果。
    result.boxes已经是NMS后的原图坐标xyxy，不能再按ONNX raw输出转置解析。
    """
    img_h, img_w = img_shape
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

    pred_boxes = []
    for box, score, cls_id in zip(xyxy, conf, cls):
        if float(score) < CONF_THRES:
            continue
        pred_boxes.append({
            "class": int(cls_id),
            "bbox": clip_box_xyxy(tuple(box.tolist()), img_w, img_h),
            "confidence": float(score),
        })
    return pred_boxes


def letterbox_bgr_for_onnx(img_bgr: np.ndarray, new_shape: int = IMG_SIZE) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    模拟Ultralytics letterbox预处理。
    返回ONNX输入、缩放比例gain、padding(dw, dh)。输入图片为cv2读取的BGR，ONNX输入转为RGB。
    """
    shape = img_bgr.shape[:2]  # h, w
    h0, w0 = shape
    gain = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * gain)), int(round(h0 * gain)))  # w, h

    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr.copy()

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, gain, (dw, dh)


def scale_boxes_to_original(
        boxes_xyxy: np.ndarray,
        orig_shape: tuple[int, int],
        gain: float,
        pad: tuple[float, float]
) -> np.ndarray:
    """将letterbox输入尺度上的xyxy框还原到原图尺度。"""
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4)

    boxes = boxes_xyxy.astype(np.float32).copy()
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= gain

    img_h, img_w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h - 1)
    return boxes


def extract_onnx_pred_boxes(
        raw_output: np.ndarray,
        orig_shape: tuple[int, int],
        gain: float,
        pad: tuple[float, float],
        conf_thres: float = CONF_THRES,
        iou_thres: float = IOU_THRES,
) -> list[dict]:
    """
    解析YOLOv8/YOLO11 ONNX raw输出。
    常见输出: (1, 4 + nc, anchors)，例如单类别为 (1, 5, 6069)。
    处理流程: squeeze -> 转置到(N, 4+nc) -> xywh转xyxy -> NMS -> 还原原图坐标。
    """
    pred = np.asarray(raw_output)
    if pred.size == 0:
        return []

    # 去掉batch维度: (1, C, N) -> (C, N)，或 (1, N, C) -> (N, C)
    while pred.ndim > 2 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim != 2:
        pred = pred.reshape(-1, pred.shape[-1])

    # YOLO导出通常是(C, N)，需要转成(N, C)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] < 5:
        return []

    boxes_xywh = pred[:, :4].astype(np.float32)
    class_part = pred[:, 4:].astype(np.float32)

    # YOLOv8/YOLO11检测头没有单独objectness，class_part即类别分数。
    # 单类别时 class_part.shape[1] == 1。
    if class_part.shape[1] == 1:
        scores = class_part[:, 0]
        classes = np.zeros_like(scores, dtype=np.int32)
    else:
        classes = np.argmax(class_part, axis=1).astype(np.int32)
        scores = class_part[np.arange(class_part.shape[0]), classes]

    keep = scores >= conf_thres
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    classes = classes[keep]

    x_c = boxes_xywh[:, 0]
    y_c = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    boxes_xyxy = np.stack([
        x_c - w / 2,
        y_c - h / 2,
        x_c + w / 2,
        y_c + h / 2,
    ], axis=1)

    keep_idx = nms_numpy(boxes_xyxy, scores, iou_thres)
    if keep_idx.size == 0:
        return []

    boxes_xyxy = boxes_xyxy[keep_idx]
    scores = scores[keep_idx]
    classes = classes[keep_idx]

    boxes_xyxy = scale_boxes_to_original(boxes_xyxy, orig_shape, gain, pad)
    img_h, img_w = orig_shape

    pred_boxes = []
    for box, score, cls_id in zip(boxes_xyxy, scores, classes):
        pred_boxes.append({
            "class": int(cls_id),
            "bbox": clip_box_xyxy(tuple(box.tolist()), img_w, img_h),
            "confidence": float(score),
        })
    return pred_boxes


# ================= 6. 匹配与可视化 =================
def match_boxes_iou(
        gt_boxes: list[dict],
        pred_boxes: list[dict],
        iou_thresh: float = IOU_MATCH_THRES,
        match_class: bool = MATCH_CLASS,
) -> dict:
    matched_gt = set()
    matched_pred = set()
    matches = []

    pred_order = sorted(
        range(len(pred_boxes)),
        key=lambda idx: pred_boxes[idx].get("confidence", 0.0),
        reverse=True,
    )

    for pred_idx in pred_order:
        pred = pred_boxes[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if match_class and int(pred["class"]) != int(gt["class"]):
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            matches.append({"gt_idx": best_gt_idx, "pred_idx": pred_idx, "iou": best_iou})

    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
    unmatched_pred = [idx for idx in range(len(pred_boxes)) if idx not in matched_pred]
    return {
        "tp": len(matches),
        "fp": len(unmatched_pred),
        "fn": len(unmatched_gt),
        "matches": matches,
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
    }


def draw_iou_match_image(
        img_path: Path,
        gt_boxes: list[dict],
        pred_boxes: list[dict],
        match_info: dict,
        save_path: Path,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return

    matched_gt = match_info["matched_gt"]
    matched_pred = match_info["matched_pred"]

    # GT标注：绿色=已匹配，黄色=漏检
    for idx, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = gt["bbox"]
        color = (0, 200, 0) if idx in matched_gt else (0, 255, 255)
        label = f"GT cls{gt['class']} ok" if idx in matched_gt else f"GT cls{gt['class']} FN"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 预测标注：蓝色=正确检测，红色=误检
    for idx, pred in enumerate(pred_boxes):
        x1, y1, x2, y2 = pred["bbox"]
        conf = pred.get("confidence", 0.0)
        color = (255, 0, 0) if idx in matched_pred else (0, 0, 255)
        label = f"TP c{pred['class']} {conf:.2f}" if idx in matched_pred else f"FP c{pred['class']} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, min(img.shape[0] - 5, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


# ================= 7. 漏检误检图片收集（PyTorch/ONNX共用）=================
def collect_issue_images(
        model_type: str,
        eval_dir: Path,
        model: Optional[YOLO] = None,
        ort_sess: Optional[ort.InferenceSession] = None,
) -> tuple[dict, list[dict]]:
    issue_root = eval_dir / f"issue_analysis_{model_type}"
    fn_dir = issue_root / "漏检图片"
    fp_dir = issue_root / "误检图片"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    fn_image_count = 0
    fp_image_count = 0
    rows = []
    speed_list = []

    img_paths = get_test_images()
    if not img_paths:
        print(f"⚠️ 测试图片目录为空或不存在: {TEST_IMG_DIR}")

    input_name = None
    if model_type == "onnx":
        if ort_sess is None:
            raise ValueError("ONNX评估需要传入ort_sess")
        input_name = ort_sess.get_inputs()[0].name
        print(f"  ONNX input name: {input_name}")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 读取图片失败，跳过: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = load_gt_boxes(img_path)
        gt_count = len(gt_boxes)

        t0 = time.perf_counter()
        if model_type == "pytorch":
            if model is None:
                raise ValueError("PyTorch评估需要传入model")
            result = model(img, conf=CONF_THRES, iou=IOU_THRES, imgsz=IMG_SIZE, verbose=False)[0]
            pred_boxes = extract_pytorch_pred_boxes(result, (img_h, img_w))
        elif model_type == "onnx":
            img_input, gain, pad = letterbox_bgr_for_onnx(img, IMG_SIZE)
            raw_output = ort_sess.run(None, {input_name: img_input})[0]
            pred_boxes = extract_onnx_pred_boxes(raw_output, (img_h, img_w), gain, pad)
        else:
            raise ValueError(f"未知model_type: {model_type}")
        t1 = time.perf_counter()

        speed_list.append((t1 - t0) * 1000)

        pred_count = len(pred_boxes)
        match_info = match_boxes_iou(gt_boxes, pred_boxes, IOU_MATCH_THRES, MATCH_CLASS)
        tp = match_info["tp"]
        fp = match_info["fp"]
        fn = match_info["fn"]

        total_gt += gt_count
        total_pred += pred_count
        total_tp += tp
        total_fp += fp
        total_fn += fn

        has_fn = fn > 0
        has_fp = fp > 0

        if has_fn:
            fn_image_count += 1
            draw_iou_match_image(img_path, gt_boxes, pred_boxes, match_info, fn_dir / img_path.name)
        if has_fp:
            fp_image_count += 1
            draw_iou_match_image(img_path, gt_boxes, pred_boxes, match_info, fp_dir / img_path.name)

        rows.append({
            "filename": img_path.name,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "has_fn": int(has_fn),
            "has_fp": int(has_fp),
        })

    manual_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    manual_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    manual_f1 = (2 * manual_precision * manual_recall) / (manual_precision + manual_recall) if (manual_precision + manual_recall) > 0 else 0.0
    avg_speed = float(np.mean(speed_list)) if speed_list else 0.0

    summary = {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "manual_tp": total_tp,
        "manual_fp": total_fp,
        "manual_fn": total_fn,
        "manual_precision": manual_precision,
        "manual_recall": manual_recall,
        "manual_f1": manual_f1,
        "fn_images": fn_image_count,
        "fp_images": fp_image_count,
        "avg_speed_ms": avg_speed,
    }

    csv_path = issue_root / "单张图片检测统计.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "gt_count", "pred_count", "tp", "fp", "fn", "has_fn", "has_fp"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return summary, rows


# ================= 8. PyTorch模型评估 =================
def evaluate_pytorch_model(model_name: str, model_path: str, batch_root: Path) -> dict:
    weight_path = Path(model_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    print(f"\n{'=' * 70}")
    print(f"评估PyTorch模型: {model_name}")
    print(f"权重路径: {weight_path}")

    model = YOLO(str(weight_path))
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    metrics = model.val(
        data=YAML_PATH,
        split="test",
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        project=str(batch_root),
        name=model_name.replace(" ", "_"),
        exist_ok=True,
        plots=True,
        cache=False,
        workers=0,
    )

    eval_dir = Path(metrics.save_dir)
    results_dict = metrics.results_dict
    precision = float(results_dict.get("metrics/precision(B)", 0.0))
    recall = float(results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n🔎 正在收集PyTorch漏检误检图片...")
    issue_summary, _ = collect_issue_images("pytorch", eval_dir=eval_dir, model=model)

    print("\n📊 官方标准指标")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1-score:     {f1:.4f}")
    print(f"  mAP@0.5:      {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    print(f"  模型参数量:   {params:.3f} M")

    print("\n📊 手动IoU匹配指标")
    print(f"  GT目标总数:   {issue_summary['total_gt']}")
    print(f"  预测目标总数: {issue_summary['total_pred']}")
    print(f"  正确检测(TP): {issue_summary['manual_tp']}")
    print(f"  误检(FP):     {issue_summary['manual_fp']}")
    print(f"  漏检(FN):     {issue_summary['manual_fn']}")
    print(f"  Precision:    {issue_summary['manual_precision']:.4f}")
    print(f"  Recall:       {issue_summary['manual_recall']:.4f}")
    print(f"  F1-score:     {issue_summary['manual_f1']:.4f}")
    print(f"  漏检图片数:   {issue_summary['fn_images']}")
    print(f"  误检图片数:   {issue_summary['fp_images']}")

    print("\n⚡ 推理速度")
    print(f"  平均单帧时延: {issue_summary['avg_speed_ms']:.2f} ms")

    summary = {
        "model": model_name,
        "weight": str(weight_path),
        "eval_dir": str(eval_dir),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "params_m": params,
        "manual_tp": issue_summary["manual_tp"],
        "manual_fp": issue_summary["manual_fp"],
        "manual_fn": issue_summary["manual_fn"],
        "manual_precision": issue_summary["manual_precision"],
        "manual_recall": issue_summary["manual_recall"],
        "manual_f1": issue_summary["manual_f1"],
        "fn_images": issue_summary["fn_images"],
        "fp_images": issue_summary["fp_images"],
        "speed_total": issue_summary["avg_speed_ms"],
    }

    print(f"\n✅ {model_name} 评估完成")
    return summary


# ================= 9. ONNX模型评估 =================
def evaluate_onnx_model(model_name: str, onnx_path: str, batch_root: Path) -> dict:
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

    print(f"\n{'=' * 70}")
    print(f"评估ONNX模型: {model_name}")
    print(f"模型路径: {onnx_path}")

    ort_sess = create_ort_session(onnx_path)

    eval_dir = batch_root / model_name.replace(" ", "_")
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔎 正在收集ONNX漏检误检图片...")
    issue_summary, _ = collect_issue_images("onnx", eval_dir=eval_dir, ort_sess=ort_sess)

    map50 = 0.0
    map50_95 = 0.0
    if RUN_ULTRALYTICS_ONNX_VAL:
        try:
            print("\n🔎 正在执行Ultralytics官方ONNX val...")
            yolo_onnx = YOLO(str(onnx_path))
            metrics = yolo_onnx.val(
                data=YAML_PATH,
                split="test",
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device=DEVICE if "CUDAExecutionProvider" in ort_sess.get_providers() else "cpu",
                project=str(batch_root),
                name=f"{model_name.replace(' ', '_')}_ultralytics_val",
                exist_ok=True,
                plots=True,
                cache=False,
                workers=0,
            )
            results_dict = metrics.results_dict
            map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
        except Exception as exc:
            print(f"⚠️ Ultralytics官方ONNX val失败，跳过: {exc}")

    print("\n📊 ONNX手动IoU匹配指标")
    print(f"  GT目标总数:   {issue_summary['total_gt']}")
    print(f"  预测目标总数: {issue_summary['total_pred']}")
    print(f"  正确检测(TP): {issue_summary['manual_tp']}")
    print(f"  误检(FP):     {issue_summary['manual_fp']}")
    print(f"  漏检(FN):     {issue_summary['manual_fn']}")
    print(f"  漏检图片数:   {issue_summary['fn_images']}")
    print(f"  误检图片数:   {issue_summary['fp_images']}")
    print(f"  Precision:    {issue_summary['manual_precision']:.4f}")
    print(f"  Recall:       {issue_summary['manual_recall']:.4f}")
    print(f"  F1-score:     {issue_summary['manual_f1']:.4f}")

    print("\n⚡ 推理速度")
    print(f"  平均单帧时延: {issue_summary['avg_speed_ms']:.2f} ms")

    summary = {
        "model": model_name,
        "weight": str(onnx_path),
        "eval_dir": str(eval_dir),
        "precision": issue_summary["manual_precision"],
        "recall": issue_summary["manual_recall"],
        "f1": issue_summary["manual_f1"],
        "map50": map50,
        "map50_95": map50_95,
        "params_m": 0.0,
        "manual_tp": issue_summary["manual_tp"],
        "manual_fp": issue_summary["manual_fp"],
        "manual_fn": issue_summary["manual_fn"],
        "manual_precision": issue_summary["manual_precision"],
        "manual_recall": issue_summary["manual_recall"],
        "manual_f1": issue_summary["manual_f1"],
        "fn_images": issue_summary["fn_images"],
        "fp_images": issue_summary["fp_images"],
        "speed_total": issue_summary["avg_speed_ms"],
    }

    del ort_sess
    print(f"\n✅ {model_name} 评估完成")
    return summary


# ================= 10. 结果保存与对比 =================
def save_comparison_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "rank", "model", "precision", "recall", "f1", "map50", "map50_95", "params_m",
        "manual_tp", "manual_fp", "manual_fn", "manual_precision", "manual_recall", "manual_f1",
        "fn_images", "fp_images", "speed_total", "weight", "eval_dir",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            filtered_row = {k: row.get(k, "") for k in fieldnames}
            filtered_row["rank"] = index
            writer.writerow(filtered_row)
    print(f"\n📄 完整对比表格已保存至: {output_path}")


def print_comparison_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 130)
    print("📊 PyTorch vs ONNX 统一对比结果")
    print("=" * 130)
    print(
        f"{'Rank':<4} {'Model':<30} {'Precision':<10} {'Recall':<8} {'F1':<8} "
        f"{'TP':<6} {'FP':<6} {'FN':<6} {'漏检图':<8} {'误检图':<8} {'速度(ms)':<10}"
    )
    print("-" * 130)

    for idx, row in enumerate(summaries, 1):
        print(
            f"{idx:<4} {row['model']:<30} {row['precision']:<10.4f} {row['recall']:<8.4f} {row['f1']:<8.4f} "
            f"{row['manual_tp']:<6} {row['manual_fp']:<6} {row['manual_fn']:<6} "
            f"{row['fn_images']:<8} {row['fp_images']:<8} {row['speed_total']:<10.2f}"
        )

    print("=" * 130)

    print("\n📈 同模型不同格式性能对比")
    print(f"  {'模型':<25} {'格式':<10} {'Recall':<10} {'Precision':<10} {'F1':<10} {'速度(ms)':<10} {'速度提升':<10}")
    print("-" * 80)

    for model_base in ["YOLO11n-CT", "YOLO11n-Normal"]:
        pt_model = next((s for s in summaries if model_base in s["model"] and "PyTorch" in s["model"]), None)
        onnx_model = next((s for s in summaries if model_base in s["model"] and "ONNX" in s["model"]), None)
        if pt_model and onnx_model:
            if pt_model["speed_total"] > 0:
                speed_improve = ((pt_model["speed_total"] - onnx_model["speed_total"]) / pt_model["speed_total"]) * 100
                speed_text = f"{speed_improve:+.2f}%"
            else:
                speed_text = "N/A"
            print(
                f"  {model_base:<25} {'PyTorch':<10} {pt_model['recall']:<10.4f} {pt_model['precision']:<10.4f} "
                f"{pt_model['f1']:<10.4f} {pt_model['speed_total']:<10.2f} {'-':<10}"
            )
            print(
                f"  {'':<25} {'ONNX':<10} {onnx_model['recall']:<10.4f} {onnx_model['precision']:<10.4f} "
                f"{onnx_model['f1']:<10.4f} {onnx_model['speed_total']:<10.2f} {speed_text:>10}"
            )
            print("-" * 80)


# ================= 11. 主函数 =================
def main() -> None:
    print_runtime_info()

    yaml_path = Path(YAML_PATH)
    test_img_dir = Path(TEST_IMG_DIR)
    test_lbl_dir = Path(TEST_LBL_DIR)

    missing_paths = [str(path) for path in (yaml_path, test_img_dir, test_lbl_dir) if not path.exists()]
    if missing_paths:
        print("❌ 缺少必要路径:")
        for path in missing_paths:
            print(f"  - {path}")
        return

    if CLEAR_LABEL_CACHE:
        clear_label_cache()

    batch_root = Path(OUTPUT_ROOT) / f"batch_eval_onnx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_root.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 所有结果将保存至: {batch_root}")

    summaries = []
    failures = []

    if AUTO_EXPORT_ONNX:
        print("\n🚀 自动导出ONNX模型...")
        for model_name, pt_path in PYTORCH_MODEL_PATHS.items():
            try:
                model = YOLO(pt_path)
                onnx_path = model.export(
                    format="onnx",
                    imgsz=IMG_SIZE,
                    dynamic=False,
                    opset=17,
                    simplify=True,
                    device=DEVICE,
                    verbose=False,
                )
                base_name = model_name.replace(" (PyTorch)", "")
                ONNX_MODEL_PATHS[f"{base_name} (ONNX)"] = str(onnx_path)
                print(f"  ✅ {base_name} 导出成功: {onnx_path}")
            except Exception as exc:
                print(f"  ❌ {model_name} 导出失败: {exc}")

    print("\n🚀 开始评估PyTorch模型...")
    for model_name, model_path in PYTORCH_MODEL_PATHS.items():
        try:
            summaries.append(evaluate_pytorch_model(model_name, model_path, batch_root))
        except Exception as exc:
            failures.append((model_name, str(exc)))
            print(f"❌ {model_name} 评估失败: {exc}")

    print("\n🚀 开始评估ONNX模型...")
    for model_name, onnx_path in ONNX_MODEL_PATHS.items():
        try:
            summaries.append(evaluate_onnx_model(model_name, onnx_path, batch_root))
        except Exception as exc:
            failures.append((model_name, str(exc)))
            print(f"❌ {model_name} 评估失败: {exc}")

    if not summaries:
        print("❌ 没有模型评估成功")
        return

    summaries.sort(key=lambda x: x["f1"], reverse=True)

    csv_path = batch_root / "PyTorch_vs_ONNX_对比结果.csv"
    save_comparison_csv(summaries, csv_path)
    print_comparison_table(summaries)

    if failures:
        print("\n⚠️ 评估失败的模型:")
        for name, err in failures:
            print(f"  - {name}: {err}")

    print(f"\n🎉 所有评估任务完成！完整结果目录: {batch_root}")
    print("💡 漏检/误检图片在对应模型目录下的 issue_analysis_* 文件夹中")


if __name__ == "__main__":
    print("=" * 80)
    print("PyTorch vs ONNX 部署性能对比评估脚本 - fixed")
    print(f"测试集: {TEST_IMG_DIR}")
    print(f"测试集图片数: {count_test_images()}")
    print(f"置信度阈值: {CONF_THRES}")
    print(f"NMS阈值: {IOU_THRES}")
    print(f"IoU匹配阈值: {IOU_MATCH_THRES}")
    print("=" * 80)

    main()
