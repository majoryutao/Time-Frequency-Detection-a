#推理
import os
import torch
from ultralytics import YOLO

# ================= 1. 路径配置 =================
MODEL_PATH = '/root/root/output/yolo11n_aug_v12/weights/best.pt'
YAML_PATH = '/root/root/split/data.yaml'  # 必须包含 test 路径
OUTPUT_ROOT = '/root/root/output/test_results'


def main():
    # 检查必要文件
    if not os.path.exists(MODEL_PATH) or not os.path.exists(YAML_PATH):
        print("❌ 错误: 请检查权重文件或 data.yaml 路径！")
        return

    # 1. 加载模型
    print(f"🚀 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2. 执行测试集评估 (这是获取指标 mAP, P, R 的核心)
    # 它会自动对比 images/test 和 labels/test
    print("📊 正在运行测试集指标评估...")
    metrics = model.val(
        data=YAML_PATH,
        split='test',  # 指定使用测试集
        imgsz=544,
        batch=16,
        conf=0.25,  # 推理置信度
        iou=0.45,  # NMS 阈值
        device=0,
        project=OUTPUT_ROOT,
        name='test_eval',  # 自动生成 test_eval, test_eval2...
        exist_ok=False,  # 自动递增，防止覆盖指标结果
        save_json=True,  # 保存结果到 json
        plots=True  # 生成 PR 曲线和混淆矩阵
    )

    # 3. 提取关键指标
    # 这里的 metrics.results_dict 包含了所有计算出的分数
    map50 = metrics.results_dict['metrics/mAP50(B)']
    map50_95 = metrics.results_dict['metrics/mAP50-95(B)']
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']

    print("\n" + "=" * 60)
    print(f"✅ 测试集评估完成！指标如下：")
    print(f"🔹 精确度 Precision: {precision:.4f}")
    print(f"🔹 召回率 Recall:    {recall:.4f}")
    print(f"🔹 mAP50:           {map50:.4f}")
    print(f"🔹 mAP50-95:        {map50_95:.4f}")

    # 获取本次结果保存的实际目录
    actual_dir = metrics.save_dir
    print(f"\n📁 详细报告(含对比图)保存在: {actual_dir}")
    print("=" * 60)

    # 4. (可选) 如果你还需要单独保存带框的预测图，可以保留这一步
    # 但其实 model.val 已经会自动在 save_dir 下生成 val_batch_pred.jpg 展示对比效果
    # print("\n🖼️ 正在生成单图推理可视化...")
    # model.predict(source='/root/root/split/images/test', project=OUTPUT_ROOT, name='visuals', save=True)


if __name__ == '__main__':
    main()