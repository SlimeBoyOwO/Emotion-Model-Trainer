import os
from train.config import Config
import numpy as np

# 固定随机种子保证可复现
SEED = Config.SEED
np.random.seed(SEED)

def main():
    """主函数：加载数据、训练和评估模型"""
    print("=== 情绪分类系统 ===")

    # 选择运行模式
    mode = input("请选择模式 (1: 训练, 2: 导出ONNX, 3: 推理, 4: 评测): ").strip()

    if mode == "1":
        print("=== 19类情绪分类模型训练 ===")
        import torch
        torch.manual_seed(SEED)
        
        from train import load_data, train_and_evaluate
        # 1. 加载数据
        train_texts, test_texts, train_labels, test_labels, label_encoder = load_data()

        # 2. 训练和评估模型
        train_and_evaluate(train_texts, test_texts, train_labels, test_labels, label_encoder)

    elif mode == "2":
        print("=== 导出ONNX模型 ===")
        from train.config import Config
        from train.onnx_exporter import convert_to_onnx
        convert_to_onnx(Config.FINAL_MODEL_DIR, Config.FINAL_MODEL_DIR + "/model.onnx", max_length=128)

    elif mode == "3":
        print("=== 情绪推理 ===")
        from inference.emotion_predictor import EmotionPredictor
        from inference.config import InferenceConfig

        # 初始化预测器
        print(f"正在加载模型: {InferenceConfig.MODEL_DIR}")
        predictor = EmotionPredictor(
            model_dir=InferenceConfig.MODEL_DIR,
            max_length=InferenceConfig.MAX_LENGTH
        )

        # 交互式推理
        print("\n输入文本进行情绪预测（输入 'quit' 退出）")
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() == "quit":
                break

            if not text:
                print("输入不能为空！")
                continue

            # 预测情绪
            emotion = predictor.predict(text, return_top_k=1)
            print(f"预测情绪: {emotion}")

            # 可选：显示top-3结果
            top_3 = predictor.predict(text, return_top_k=3)
            print("Top-3 可能的情绪:")
            for label, prob in top_3:
                print(f"  {label}: {prob*100:.2f}%")

    elif mode == "4":
        print("=== 模型评测模式 ===")

        from train.config import Config

        # 获取测试数据集路径
        test_csv_path = os.path.join(Config.DATA_DIR, "test.csv")
        if not os.path.exists(test_csv_path):
            print(f"错误: 测试数据集不存在: {test_csv_path}")
            return
        
        # 选择模型类型
        print("支持的模型类型:")
        print("1. ONNX (model.onnx)")
        print("2. Safetensors (model.safetensors)")
        print("3. PyTorch (pytorch_model.bin)")
        
        model_type_choice = input("请选择模型类型 (1/2/3): ").strip()
        model_type_map = {"1": "onnx", "2": "safetensors", "3": "pytorch"}
        
        if model_type_choice not in model_type_map:
            print("无效的选择！")
            return
        
        model_type = model_type_map[model_type_choice]
        
        # 获取模型路径
        model_path = Config.FINAL_MODEL_DIR
        print(f"使用模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if model_type == "onnx":
            model_file = os.path.join(model_path, "model.onnx")
        elif model_type == "safetensors":
            import torch
            torch.manual_seed(SEED)
            model_file = os.path.join(model_path, "model.safetensors")
        else:
            import torch
            torch.manual_seed(SEED)
            model_file = os.path.join(model_path, "pytorch_model.bin")
        
        if not os.path.exists(model_file):
            print(f"错误: 模型文件不存在: {model_file}")
            print("请确保模型文件存在，或者先运行训练模式(1)生成模型。")
            return
        
        print(f"\n开始评测 {model_type} 模型...")
        print(f"测试数据集: {test_csv_path}")
        try:
            from train.evaluation import evaluate_model
            results = evaluate_model(model_path, model_type, test_csv_path)
            print("\n评测完成！")
        except Exception as e:
            print(f"评测失败: {e}")

    else:
        print("无效的模式选择！")

if __name__ == "__main__":
    main()