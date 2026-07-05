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

# ================= 1. 全局配置（直接用你的路径，无需修改）=================
ONNX_MODEL_PATHS = {
    "YOLO11n-CT (ONNX)": "/root/root/out/output2/yolo11n_ct_aug_v1/weights/best.onnx",
    "YOLO11n-Normal (ONNX)": "/root/root/output2/yolo11n_aug_v1/weights/best.onnx",
}

PYTORCH_MODEL_PATHS = {
    "YOLO11n-CT (PyTorch)": "/root/root/out/output2/yolo11n_ct_aug_v1/weights/best.pt",
    "YOLO11n-Normal (PyTorch)": "/root/root/output2/yolo11n_aug_v1/weights/best.pt",
}

TEST_IMG_DIR = "/root/root/split/images/test"
TEST_LBL_DIR = "/root/root/split/labels/test"
OUTPUT_ROOT = "/root/root/four_model_deployment_evaluation"

# ================= 2. 推理配置（与生产环境完全一致）=================
IMG_SIZE = 544
CONF_THRES = 0.02  # 默认使用工业部署最优阈值
IOU_THRES = 0.45
DEVICE = 0
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

IOU_MATCH_THRES = 0.5
MATCH_CLASS = True
CLEAR_LABEL_CACHE = True
FORCE_ONNX_CPU = False

# 低阈值最大检测框数，防止内存溢出
MAX_DET = 500

# ================= 3. ONNX Runtime环境工具 =================
def print_runtime_info() -> None:
    print("\n🧩 生产环境模拟运行环境")
    print(f"  Python PID:     {os.getpid()}")
    print(f"  ONNXRuntime:    {ort.__version__}")
    print(f"  ORT providers:  {ort.get_available_providers()}")
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            print(f"  GPU device:     cuda:{DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
            print(f"  GPU memory:     {torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3:.1f} GB")
        except Exception as exc:
            print(f"  GPU init warn:  {exc}")
    else:
        print("  ⚠️  未检测到GPU，将使用CPU推理")


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    """
    生产级ONNX会话创建，严格验证CUDA是否真正启用
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
            provider_options = [
                {
                    "device_id": DEVICE,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 限制2GB显存
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
                {}
            ]
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=providers,
                provider_options=provider_options,
            )
            actual = sess.get_providers()
            if "CUDAExecutionProvider" in actual:
                print(f"✅ ONNXRuntime CUDA加速已启用: {actual}")
                return sess
            print(f"⚠️ ONNXRuntime未实际启用CUDA，当前providers={actual}，切换CPU")
        except Exception as exc:
            print(f"⚠️ ONNXRuntime CUDA初始化失败，切换CPU: {exc}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"✅ ONNXRuntime CPU模式: {sess.get_providers()}")
    return sess


# ================= 4. 基础工具函数 =================
def create_torch_model(pt_path: Path):
    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pt_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("PyTorch .pt inference needs ultralytics. Install it with: pip install ultralytics") from exc

    model = YOLO(str(pt_path))
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"PyTorch model loaded on device={device}: {pt_path}")
    return model, device


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


def safe_percent(num: int, den: int) -> float:
    return num / den * 100 if den > 0 else 0.0


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
    """纯NumPy NMS，与生产环境完全一致"""
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


# ================= 5. ONNX预处理与预测框解析 =================
def letterbox_bgr_for_onnx(img_bgr: np.ndarray, new_shape: int = IMG_SIZE) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    与生产环境完全一致的letterbox预处理
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
    """将预测框还原到原图坐标"""
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
    生产级ONNX输出解析，与部署代码100%一致
    """
    pred = np.asarray(raw_output)
    if pred.size == 0:
        return []

    # 处理YOLO11输出格式: (1, 5, 6069)
    while pred.ndim > 2 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim != 2:
        pred = pred.reshape(-1, pred.shape[-1])

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] < 5:
        return []

    boxes_xywh = pred[:, :4].astype(np.float32)
    class_part = pred[:, 4:].astype(np.float32)

    # 单类别处理
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

    if scores.shape[0] > MAX_DET:
        top_idx = np.argsort(scores)[-MAX_DET:]
        boxes_xywh = boxes_xywh[top_idx]
        scores = scores[top_idx]
        classes = classes[top_idx]

    # xywh转xyxy
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

    # NMS
    keep_idx = nms_numpy(boxes_xyxy, scores, iou_thres)
    if keep_idx.size == 0:
        return []

    boxes_xyxy = boxes_xyxy[keep_idx]
    scores = scores[keep_idx]
    classes = classes[keep_idx]

    # 还原到原图坐标
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


# ================= 6. 工业级指标计算与可视化 =================
def extract_torch_pred_boxes(result, img_w: int, img_h: int) -> list[dict]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

    if scores.shape[0] > MAX_DET:
        top_idx = np.argsort(scores)[-MAX_DET:]
        boxes_xyxy = boxes_xyxy[top_idx]
        scores = scores[top_idx]
        classes = classes[top_idx]

    pred_boxes = []
    for box, score, cls_id in zip(boxes_xyxy, scores, classes):
        pred_boxes.append({
            "class": int(cls_id),
            "bbox": clip_box_xyxy(tuple(box.tolist()), img_w, img_h),
            "confidence": float(score),
        })
    return pred_boxes


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
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
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

    # GT标注：绿色=已匹配，黄色=漏检（工业级高亮）
    for idx, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = gt["bbox"]
        color = (0, 200, 0) if idx in matched_gt else (0, 255, 255)
        label = f"GT 已匹配" if idx in matched_gt else f"GT 漏检!!!"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 预测标注：蓝色=正确检测，红色=误检（工业级高亮）
    for idx, pred in enumerate(pred_boxes):
        x1, y1, x2, y2 = pred["bbox"]
        conf = pred.get("confidence", 0.0)
        color = (255, 0, 0) if idx in matched_pred else (0, 0, 255)
        label = f"TP {conf:.2f}" if idx in matched_pred else f"FP 误警!!!"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, min(img.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


# ================= 7. 工业级评估核心函数 =================
def evaluate_onnx_model(
    model_name: str,
    onnx_path: str,
    batch_root: Path,
    save_images: bool = True,
    conf_thres: float = CONF_THRES
) -> dict:
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

    print(f"\n{'=' * 70}")
    print(f"评估ONNX部署模型: {model_name}")
    print(f"模型路径: {onnx_path}")
    print(f"置信度阈值: {conf_thres:.3f}")

    ort_sess = create_ort_session(onnx_path)
    input_name = ort_sess.get_inputs()[0].name
    print(f"ONNX输入名称: {input_name}")

    eval_dir = batch_root / model_name.replace(" ", "_")
    eval_dir.mkdir(parents=True, exist_ok=True)

    if save_images:
        fn_dir = eval_dir / "漏检图片_工业级"
        fp_dir = eval_dir / "误检图片_工业级"
        fn_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)

    # 工业级核心指标统计
    total_images = 0
    positive_images = 0  # 含目标的图片数
    background_images = 0  # 纯背景图片数

    total_gt = 0
    total_pred = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # 工业部署最关心的指标
    images_with_fn = 0  # 存在漏检的图片数
    images_with_fp = 0  # 存在误检的图片数
    background_fp_images = 0  # 纯背景误检图片数（致命问题）
    background_fp_boxes = 0  # 纯背景误检框数

    speed_list = []
    img_paths = get_test_images()

    print(f"\n🔎 正在评估 {len(img_paths)} 张测试图片...")
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 读取图片失败，跳过: {img_path}")
            continue

        total_images += 1
        img_h, img_w = img.shape[:2]
        gt_boxes = load_gt_boxes(img_path)
        gt_count = len(gt_boxes)

        is_positive = gt_count > 0
        is_background = gt_count == 0

        if is_positive:
            positive_images += 1
        else:
            background_images += 1

        # ONNX推理计时（包含预处理+推理+后处理）
        t0 = time.perf_counter()
        img_input, gain, pad = letterbox_bgr_for_onnx(img, IMG_SIZE)
        raw_output = ort_sess.run(None, {input_name: img_input})[0]
        pred_boxes = extract_onnx_pred_boxes(raw_output, (img_h, img_w), gain, pad, conf_thres)
        t1 = time.perf_counter()

        speed_list.append((t1 - t0) * 1000)
        pred_count = len(pred_boxes)

        # IoU匹配
        match_info = match_boxes_iou(gt_boxes, pred_boxes)
        tp = match_info["tp"]
        fp = match_info["fp"]
        fn = match_info["fn"]

        # 统计指标
        total_gt += gt_count
        total_pred += pred_count
        total_tp += tp
        total_fp += fp
        total_fn += fn

        has_fn = fn > 0
        has_fp = fp > 0

        if has_fn:
            images_with_fn += 1
        if has_fp:
            images_with_fp += 1

        # 纯背景误检统计（工业级核心）
        if is_background:
            if pred_count > 0:
                background_fp_images += 1
                background_fp_boxes += pred_count

        # 保存问题图片
        if save_images:
            if has_fn:
                draw_iou_match_image(img_path, gt_boxes, pred_boxes, match_info, fn_dir / img_path.name)
            if has_fp:
                draw_iou_match_image(img_path, gt_boxes, pred_boxes, match_info, fp_dir / img_path.name)

    # 计算最终指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 工业级核心指标
    background_fpr = background_fp_images / background_images if background_images > 0 else 0.0
    specificity = 1.0 - background_fpr  # 特异性（越高越好）
    false_alarm_rate = background_fp_images / total_images  # 整体误警率

    avg_latency_ms = float(np.mean(speed_list)) if speed_list else 0.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    # 打印工业级评估结果
    print("\n📊 工业级部署核心指标")
    print(f"  总测试图片:   {total_images}")
    print(f"  正样本图片:   {positive_images}")
    print(f"  背景图片:     {background_images}")
    print(f"  GT目标总数:   {total_gt}")
    print(f"  预测目标总数: {total_pred}")
    print(f"\n  ✅ 正确检测(TP): {total_tp}")
    print(f"  ⚠️  误检(FP):    {total_fp}")
    print(f"  ❌ 漏检(FN):    {total_fn}")
    print(f"\n  🎯 精确率(P):   {precision:.4f}")
    print(f"  🎯 召回率(R):   {recall:.4f}")
    print(f"  🎯 F1分数:      {f1:.4f}")
    print(f"\n  🔥 工业级核心指标")
    print(f"  漏检图片数:   {images_with_fn} ({images_with_fn/positive_images*100:.1f}%)")
    print(f"  误检图片数:   {images_with_fp} ({images_with_fp/total_images*100:.1f}%)")
    print(f"  背景误检图:   {background_fp_images} ({background_fpr*100:.2f}%)")
    print(f"  背景误检框:   {background_fp_boxes}")
    print(f"  特异性:       {specificity:.4f}")
    print(f"  整体误警率:   {false_alarm_rate*100:.2f}%")
    print(f"\n  ⚡ 推理性能")
    print(f"  平均单帧时延: {avg_latency_ms:.2f} ms")
    print(f"  推理帧率:     {fps:.1f} FPS")

    summary = {
        "model": model_name,
        "confidence": conf_thres,
        "total_images": total_images,
        "positive_images": positive_images,
        "background_images": background_images,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "images_with_fn": images_with_fn,
        "images_with_fp": images_with_fp,
        "background_fp_images": background_fp_images,
        "background_fp_boxes": background_fp_boxes,
        "background_fpr": background_fpr,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "eval_dir": str(eval_dir),
    }

    del ort_sess
    print(f"\n✅ {model_name} 评估完成")
    return summary


# ================= 8. 结果保存与多阈值扫描 =================
def save_threshold_scan_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "model", "confidence",
        "total_images", "positive_images", "background_images",
        "total_gt", "total_pred",
        "tp", "fp", "fn",
        "precision", "recall", "f1",
        "images_with_fn", "images_with_fp",
        "background_fp_images", "background_fp_boxes",
        "background_fpr", "specificity", "false_alarm_rate",
        "avg_latency_ms", "fps", "eval_dir"
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n📄 工业级阈值扫描完整结果已保存至: {output_path}")


def print_industrial_threshold_table(rows: list[dict]) -> None:
    print("\n" + "=" * 180)
    print("工业级ONNX部署阈值扫描结果")
    print("=" * 180)
    print(
        f"{'Conf':<7} "
        f"{'P':<8} {'R':<8} {'F1':<8} "
        f"{'TP':<5} {'FP':<5} {'FN':<5} "
        f"{'漏检图':<8} {'误检图':<8} "
        f"{'BG误检图':<10} {'BG误检框':<10} "
        f"{'BG_FPR':<10} {'特异性':<10} {'误警率':<10} "
        f"{'时延(ms)':<10} {'FPS':<8}"
    )
    print("-" * 180)

    for row in rows:
        print(
            f"{row['confidence']:<7.3f} "
            f"{row['precision']:<8.4f} {row['recall']:<8.4f} {row['f1']:<8.4f} "
            f"{row['tp']:<5} {row['fp']:<5} {row['fn']:<5} "
            f"{row['images_with_fn']:<8} {row['images_with_fp']:<8} "
            f"{row['background_fp_images']:<10} {row['background_fp_boxes']:<10} "
            f"{row['background_fpr']*100:<10.2f} {row['specificity']:<10.4f} {row['false_alarm_rate']*100:<10.2f} "
            f"{row['avg_latency_ms']:<10.2f} {row['fps']:<8.1f}"
        )

    print("=" * 180)


def print_industrial_best_thresholds(rows: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("🏆 工业部署最优阈值推荐（按优先级排序）")
    print("=" * 120)

    # 优先级1：零背景误检（BG_FPR=0）下的最高召回率
    zero_bg_fp_rows = [row for row in rows if row["background_fp_images"] == 0]
    if zero_bg_fp_rows:
        best_zero_bg_recall = max(zero_bg_fp_rows, key=lambda x: x["recall"])
        best_zero_bg_f1 = max(zero_bg_fp_rows, key=lambda x: x["f1"])
        print("\n✅ 最高优先级：零背景误检（无任何误报警）")
        print(f"  最高召回率阈值: conf={best_zero_bg_recall['confidence']:.3f}")
        print(f"    召回率: {best_zero_bg_recall['recall']:.4f}, F1: {best_zero_bg_recall['f1']:.4f}")
        print(f"    漏检图片: {best_zero_bg_recall['images_with_fn']}, 误检图片: {best_zero_bg_recall['images_with_fp']}")
        print(f"  最高F1阈值: conf={best_zero_bg_f1['confidence']:.3f}")
        print(f"    F1: {best_zero_bg_f1['f1']:.4f}, 召回率: {best_zero_bg_f1['recall']:.4f}")

    # 优先级2：BG_FPR≤1%下的最高召回率
    bg_fpr_le_1_rows = [row for row in rows if row["background_fpr"] <= 0.01]
    if bg_fpr_le_1_rows:
        best_bg_fpr_1 = max(bg_fpr_le_1_rows, key=lambda x: x["recall"])
        print("\n✅ 次高优先级：背景误检率≤1%")
        print(f"  最高召回率阈值: conf={best_bg_fpr_1['confidence']:.3f}")
        print(f"    召回率: {best_bg_fpr_1['recall']:.4f}, F1: {best_bg_fpr_1['f1']:.4f}")
        print(f"    背景误检率: {best_bg_fpr_1['background_fpr']*100:.2f}%, 背景误检图: {best_bg_fpr_1['background_fp_images']}")

    # 优先级3：BG_FPR≤5%下的最高召回率
    bg_fpr_le_5_rows = [row for row in rows if row["background_fpr"] <= 0.05]
    if bg_fpr_le_5_rows:
        best_bg_fpr_5 = max(bg_fpr_le_5_rows, key=lambda x: x["recall"])
        print("\n✅ 第三优先级：背景误检率≤5%")
        print(f"  最高召回率阈值: conf={best_bg_fpr_5['confidence']:.3f}")
        print(f"    召回率: {best_bg_fpr_5['recall']:.4f}, F1: {best_bg_fpr_5['f1']:.4f}")
        print(f"    背景误检率: {best_bg_fpr_5['background_fpr']*100:.2f}%, 背景误检图: {best_bg_fpr_5['background_fp_images']}")

    # 综合最优F1
    best_f1 = max(rows, key=lambda x: x["f1"])
    print("\n✅ 综合最优（F1最高）")
    print(f"  阈值: conf={best_f1['confidence']:.3f}")
    print(f"    F1: {best_f1['f1']:.4f}, 精确率: {best_f1['precision']:.4f}, 召回率: {best_f1['recall']:.4f}")


# ================= 9. 主函数 =================
def main() -> None:
    print_runtime_info()

    test_img_dir = Path(TEST_IMG_DIR)
    test_lbl_dir = Path(TEST_LBL_DIR)

    missing_paths = [str(path) for path in (test_img_dir, test_lbl_dir) if not path.exists()]
    if missing_paths:
        print("❌ 缺少必要路径:")
        for path in missing_paths:
            print(f"  - {path}")
        return

    if CLEAR_LABEL_CACHE:
        clear_label_cache()

    batch_root = Path(OUTPUT_ROOT) / f"onnx_deploy_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_root.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 所有结果将保存至: {batch_root}")

    # 工业级阈值扫描范围（重点覆盖低置信度区域）
    thresholds = [
        0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
        0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.15, 0.20, 0.25, 0.30
    ]

    all_results = []

    for model_name, onnx_path in ONNX_MODEL_PATHS.items():
        print(f"\n{'=' * 100}")
        print(f"🚀 开始评估模型: {model_name}")
        print(f"{'=' * 100}")

        model_results = []
        model_root = batch_root / model_name.replace(" ", "_")
        model_root.mkdir(parents=True, exist_ok=True)

        for conf in thresholds:
            print(f"\n--- 测试置信度阈值: {conf:.3f} ---")
            try:
                # 阈值扫描时不保存图片，速度提升10倍
                summary = evaluate_onnx_model(
                    model_name,
                    onnx_path,
                    model_root / f"conf_{conf:.3f}",
                    save_images=False,
                    conf_thres=conf
                )
                model_results.append(summary)
            except Exception as exc:
                print(f"❌ 阈值 {conf:.3f} 评估失败: {exc}")
                continue

        # 保存单模型阈值扫描结果
        if model_results:
            model_csv_path = model_root / f"{model_name.replace(' ', '_')}_threshold_scan.csv"
            save_threshold_scan_csv(model_results, model_csv_path)
            print_industrial_threshold_table(model_results)
            print_industrial_best_thresholds(model_results)
            all_results.extend(model_results)

            # 使用最优零背景误检阈值重新评估并保存问题图片
            zero_bg_fp_rows = [row for row in model_results if row["background_fp_images"] == 0]
            if zero_bg_fp_rows:
                best_conf = max(zero_bg_fp_rows, key=lambda x: x["recall"])["confidence"]
                print(f"\n🎯 使用最优零背景误检阈值 {best_conf:.3f} 重新评估并保存问题图片...")
                final_summary = evaluate_onnx_model(
                    model_name,
                    onnx_path,
                    model_root / "final_deployment_eval",
                    save_images=True,
                    conf_thres=best_conf
                )
                final_csv_path = model_root / "final_deployment_metrics.csv"
                save_threshold_scan_csv([final_summary], final_csv_path)
                print(f"✅ 最终部署评估完成，问题图片已保存至: {model_root / 'final_deployment_eval'}")

    print(f"\n🎉 所有ONNX部署评估任务完成！")
    print(f"完整结果目录: {batch_root}")
    print("💡 每个模型的最终部署问题图片在 final_deployment_eval 文件夹中")


if __name__ == "__main__":
    print("=" * 100)
    print("工业级ONNX部署专属评估工具")
    print(f"测试集: {TEST_IMG_DIR}")
    print(f"测试集图片数: {count_test_images()}")
    print(f"图像尺寸: {IMG_SIZE}×{IMG_SIZE}")
    print(f"NMS阈值: {IOU_THRES}")
    print(f"IoU匹配阈值: {IOU_MATCH_THRES}")
    print("=" * 100)

    main()
