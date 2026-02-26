# onnx_exporter.py
import torch
import os
from transformers import BertForSequenceClassification, BertTokenizer

def convert_to_onnx(model_dir, output_path, max_length=128):
    """
    将保存的 Transformers 模型转换为 ONNX 格式。
    
    Args:
        model_dir (str): 包含 pytorch/safetensors 模型和配置文件的目录路径
        output_path (str): ONNX 文件的保存路径 (包含文件名, 例如 model.onnx)
        max_length (int): 模型输入的最大序列长度，用于生成 dummy input
    """
    print(f"\n[ONNX] 开始转换模型...")
    print(f"[ONNX] 加载模型自: {model_dir}")

    try:
        # 1. 加载模型和分词器 (使用 CPU 进行导出即可)
        device = torch.device("cpu")
        model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        # 2. 准备虚拟输入 (Dummy Input)
        # BERT 需要 input_ids 和 attention_mask
        dummy_input_ids = torch.randint(0, 1000, (1, max_length), device=device)
        dummy_attention_mask = torch.ones((1, max_length), device=device)
        
        # 将输入打包成 tuple
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)

        # 3. 定义输入输出名称
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]

        # 4. 定义动态轴 (允许 batch_size 变化)
        dynamic_axes = {
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }

        # 5. 执行导出
        print(f"[ONNX] 正在导出到: {output_path} ...")
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=20,
            do_constant_folding=True
        )
        
        print(f"[ONNX] 转换成功！模型已保存至: {output_path}")
        
        # (可选) 验证生成的 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("[ONNX] 模型结构检查通过。")
        except ImportError:
            print("[ONNX] 未安装 'onnx' 库，跳过结构检查。")
        except Exception as e:
            print(f"[ONNX] 警告: 模型检查失败: {e}")

    except Exception as e:
        print(f"[ONNX] 转换过程中发生错误: {e}")
        raise e