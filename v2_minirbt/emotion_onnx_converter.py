import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import json
import onnxruntime as rt
import numpy as np

TARGET_EMOTIONS = [
    "兴奋", "厌恶", "害怕", "害羞", "慌张",      
    "担心", "无奈", "生气", "疑惑", "紧张",      
    "认真", "高兴", "哭泣", "心动", "调皮",      
    "难为情", "自信", "惊讶", "平静"        
]
NUM_LABELS = len(TARGET_EMOTIONS)

class AdvancedClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate=0.3):
        super().__init__()
        # 融合 CLS + Mean + Max 三种特征
        self.dense = nn.Linear(hidden_size * 3, 512)
        self.norm = nn.LayerNorm(512)
        self.activation = nn.GELU()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(8)]) # 增加 Dropout 采样数
        self.out_proj = nn.Linear(512, num_labels)
        
    def forward(self, x):
        x = self.activation(self.norm(self.dense(x)))
        logits = sum([self.out_proj(dropout(x)) for dropout in self.dropouts]) / len(self.dropouts)
        return logits

class EmotionModelForInference(nn.Module):
    """
    用于ONNX转换的推理模型版本，只返回logits
    """
    def __init__(self, model_name="hfl/minirbt-h288"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.classifier = AdvancedClassificationHead(self.config.hidden_size, NUM_LABELS)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # 1. CLS 向量
        cls_token = last_hidden[:, 0, :]
        # 2. Mean Pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean_pooled = torch.sum(last_hidden * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        # 3. Max Pooling
        max_pooled = torch.max(last_hidden + (1 - mask_expanded) * -1e9, 1)[0]
        
        # 特征拼接
        combined = torch.cat([cls_token, mean_pooled, max_pooled], dim=1)
        logits = self.classifier(combined)
        
        return logits

def convert_to_onnx(model_path, output_onnx_path, opset_version=18):
    """
    将PyTorch模型转换为ONNX格式
    """
    # 创建模型实例
    model = EmotionModelForInference()
    
    # 加载训练好的权重
    checkpoint = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()  # 设置为评估模式
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 准备示例输入
    dummy_text = "这是一个示例文本用于模型转换"
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # 导出到ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # 执行常量折叠以优化
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"模型已成功转换为ONNX格式: {output_onnx_path}")
    
    # 保存tokenizer和标签映射
    tokenizer.save_pretrained(output_onnx_path.replace('.onnx', '_tokenizer/'))
    
    mapping = {"id2label": {i: l for i, l in enumerate(TARGET_EMOTIONS)}, 
               "label2id": {l: i for i, l in enumerate(TARGET_EMOTIONS)}}
    with open(output_onnx_path.replace('.onnx', '_labels.json'), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    return model, tokenizer

def optimize_onnx_model(onnx_model_path):
    """
    使用onnxoptimizer优化ONNX模型
    """
    try:
        import onnx
        import onnxoptimizer
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_model_path)
        
        # 获取所有可用的优化
        all_passes = onnxoptimizer.get_available_passes()
        print(f"可用的优化pass: {all_passes}")

        # 选择基础的、安全的优化passes
        desired_passes = [
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_monotone_argmax',
            'eliminate_nop_pad',
            'extract_constant_to_initializer',
            'eliminate_unused_initializer'
        ]

        # 只保留可用的pass
        passes = [p for p in desired_passes if p in all_passes]
        print(f"将应用的优化pass: {passes}")
        if not passes:
            print("警告: 没有可用的优化pass，跳过优化")
            return onnx_model_path

        # 应用优化
        optimized_model = onnxoptimizer.optimize(onnx_model, passes)
        
        # 保存优化后的模型
        optimized_path = onnx_model_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)
        
        print(f"模型已优化并保存至: {optimized_path}")
        
        return optimized_path
    except ImportError:
        print("未安装onnxoptimizer，跳过优化步骤。请运行: pip install onnxoptimizer")
        return onnx_model_path

def test_onnx_model(onnx_model_path, test_text="这是一个测试句子"):
    """
    测试ONNX模型是否正常工作
    """
    # 加载ONNX Runtime推理会话
    sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # 加载tokenizer
    tokenizer_path = onnx_model_path.replace('.onnx', '_tokenizer/')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 对测试文本进行编码
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # 运行推理
    outputs = sess.run(
        None,
        {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
    )
    
    # 获取预测结果
    logits = outputs[0]
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_id].item()
    
    # 加载标签映射
    label_path = onnx_model_path.replace('.onnx', '_labels.json')
    with open(label_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    predicted_label = label_mapping['id2label'][str(predicted_class_id)]
    
    print(f"输入文本: {test_text}")
    print(f"预测情感: {predicted_label}")
    print(f"置信度: {confidence:.4f}")
    
    return predicted_label, confidence

if __name__ == "__main__":
    # 定义路径
    model_path = "./model_minirbt_final_v5"  # 训练好的模型路径
    output_onnx_path = "./emotion_model.onnx"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径 {model_path} 不存在")
        exit(1)
    
    # 转换为ONNX
    print("正在转换模型为ONNX格式...")
    pytorch_model, tokenizer = convert_to_onnx(model_path, output_onnx_path)
    
    # 优化ONNX模型
    print("正在优化ONNX模型...")
    optimized_onnx_path = optimize_onnx_model(output_onnx_path)
    
    # 测试优化后的模型
    print("正在测试ONNX模型...")
    test_onnx_model(optimized_onnx_path, "今天心情真好！")
    
    print("转换和优化完成！")