# yolo11n_ct 数据增强开启版
import os
import torch
from ultralytics import YOLO

# ================= 1. 路径配置 =================
DATA_YAML = '/root/root/split/data.yaml'
MODEL_YAML = '/root/root/split/data-CT.yaml'
PRETRAINED_PT = '/root/root/model/yolo11n.pt'
PROJECT_DIR = '/root/root/output2'

# ================= 2. 实验标签配置 =================
MODEL_TAG = 'yolo11n_ct'   # 可改为 yolo11n / yolo11n_c3x / yolo11n_ct
USE_AUG = True             # True=数据增强, False=不加数据增强
VERSION = 'v1'


def print_val_metrics(metrics):
    """打印 Precision / Recall / mAP50 / mAP50-95。"""
    try:
        p = metrics.box.mp
        r = metrics.box.mr
        map50 = metrics.box.map50
        map5095 = metrics.box.map

        print("\n================ 验证指标汇总 ================")
        print(f"Precision (P)   : {p:.4f}")
        print(f"Recall    (R)   : {r:.4f}")
        print(f"mAP@50          : {map50:.4f}")
        print(f"mAP@50-95       : {map5095:.4f}")
        print("============================================\n")
    except Exception as e:
        print(f"⚠️ 指标打印失败: {e}")


def build_run_name(model_tag, use_aug, version):
    aug_tag = 'aug' if use_aug else 'noaug'
    return f'{model_tag}_{aug_tag}_{version}'


def main():
    if not torch.cuda.is_available():
        print("❌ 警告: 系统未检测到可用 GPU！")
        return

    run_name = build_run_name(MODEL_TAG, USE_AUG, VERSION)

    print(f"🚀 准备开始训练，使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📌 数据集 YAML: {DATA_YAML}")
    print(f"📌 模型结构 YAML: {MODEL_YAML}")
    print(f"📌 预训练权重: {PRETRAINED_PT}")
    print(f"📌 实验名称: {run_name}")
    print(f"📌 数据增强状态: {'开启' if USE_AUG else '关闭'}")

    model = YOLO(MODEL_YAML).load(PRETRAINED_PT)

    print("\n📌 模型结构与规模如下（包含 Params / GFLOPs）：")
    model.info()

    # 严格区分增强/非增强
    if USE_AUG:
        train_args = dict(
            mosaic=1.0,
            mixup=0.15,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
            degrees=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.1,
            erasing=0.2,
        )
    else:
        train_args = dict(
            mosaic=0.0,
            mixup=0.0,
            scale=0.0,
            translate=0.0,
            fliplr=0.0,
            flipud=0.0,
            degrees=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            erasing=0.0,
        )

    results = model.train(
        data=DATA_YAML,
        project=PROJECT_DIR,
        name=run_name,
        exist_ok=False,

        # --- 基础参数 ---
        epochs=150,
        imgsz=544,
        batch=16,
        workers=8,
        device=0,
        amp=False,

        # --- 数据增强参数 ---
        **train_args,

        plots=True,
        val=True
    )

    actual_save_dir = results.save_dir
    print(f"\n📂 本次训练结果保存在: {actual_save_dir}")

    # 用训练后的 best.pt 做验证，避免混淆
    best_pt = os.path.join(actual_save_dir, 'weights', 'best.pt')
    eval_model = YOLO(best_pt)

    print("\n📊 正在执行详细评估分析...")
    metrics = eval_model.val(
        data=DATA_YAML,
        split='val',
        imgsz=544,
        iou=0.6,
        conf=0.25,
        device=0,
        save_json=True,
        plots=True,
        project=PROJECT_DIR,
        name=run_name,
        exist_ok=True
    )

    print_val_metrics(metrics)

    print(f"\n✅ 任务完成！当前实验: {run_name}")


if __name__ == '__main__':
    main()