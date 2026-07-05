#增加数据增强
import os
import torch
from ultralytics import YOLO

# ================= 1. 路径配置 =================
YAML_PATH = '/root/root/split/data.yaml'
MODEL_PATH = '/root/root/model/yolo11n.pt'
PROJECT_DIR = '/root/root/output2'


def main():
    if not torch.cuda.is_available():
        print("❌ 警告: 系统未检测到可用 GPU！")
        return

    print(f"🚀 准备开始训练，使用 GPU: {torch.cuda.get_device_name(0)}")

    # ================= 2. 加载并训练 =================
    model = YOLO(MODEL_PATH)

    # 1. 执行训练
    # 文件夹会自动递增 (baseline, baseline2, baseline3...)
    results = model.train(
        data=YAML_PATH,
        project=PROJECT_DIR,
        name='yolo11n_aug_v1',
        exist_ok=False,  # 确保文件夹不被覆盖，自动递增

        # --- 基础参数 ---
        epochs=150,
        imgsz=544,
        batch=16,
        workers=8,
        device=0,

        # --- 🔥 针对频谱图的数据增强配置 ---
        mosaic=1.0,  # 开启：将4张图拼在一起，增加模型对小目标的感知
        mixup=0.15,  # 开启：模拟“泄漏声+水噪”的叠加效果，增强鲁棒性
        scale=0.5,  # 开启：随机缩放，配合 imgsz 解决 4980 压缩导致的主体太小问题
        translate=0.1,  # 开启：随机平移，模拟泄漏发生的不同时间点
        fliplr=0.5,  # 开启：左右翻转（时间轴反转），通常对平稳泄漏声有效

        flipud=0.0,  # ❌ 关闭：禁止上下翻转！频率顺序不能乱
        degrees=0.0,  # ❌ 关闭：禁止旋转！频谱轴必须垂直
        hsv_h=0.0,  # ❌ 关闭：频谱图颜色通常代表能量，色调变化无意义
        hsv_s=0.0,  # ❌ 关闭
        hsv_v=0.1,  # 略微调整：模拟信号强弱（亮度）的变化

        erasing=0.2,  # 开启：随机遮挡部分频谱，模拟信号瞬时丢失
        plots=True,
        val=True
    )

    # 获取本次训练实际生成的目录
    actual_save_dir = results.save_dir
    print(f"\n📂 本次训练结果保存在: {actual_save_dir}")

    # 2. 训练后深入评估 (使用最佳权重)
    print("\n📊 正在执行详细评估分析...")
    model.val(
        split='val',
        iou=0.6,
        conf=0.25,
        save_json=True,
        plots=True,
        project=PROJECT_DIR,
        name=os.path.basename(actual_save_dir),  # 存入同一个递增文件夹
        exist_ok=True
    )

    print(f"\n✅ 任务完成！对比之前的 Baseline 文件夹，看看 Recall 是否提升了。")


if __name__ == '__main__':
    main()