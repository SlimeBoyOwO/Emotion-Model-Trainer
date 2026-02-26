import os
import json
from typing import Optional

from transformers import BertTokenizer
from .config import Config


class ModelEvaluator:
    """模型评估器，支持ONNX和safetensors/pytorch模型格式"""
    
    def __init__(self, model_path, model_type="onnx"):
        """
        初始化模型评估器
        
        Args:
            model_path (str): 模型路径
            model_type (str): 模型类型，可选 "onnx", "safetensors", "pytorch"
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.tokenizer: Optional[BertTokenizer] = None
        self.model = None
        self.id2label = {}
        self.label2id = {}
        # ONNX特定属性
        self.input_name = ""
        self.attention_mask_name = ""
        self.output_name = ""
        
        # 加载模型和配置
        self._load_model_and_config()
    
    def _load_model_and_config(self):
        """加载模型和标签映射"""
        # 加载标签映射
        label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在: {label_mapping_path}")
        
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
            self.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            self.label2id = label_mapping["label2id"]
        
        # 动态导入transformers
        from transformers import BertTokenizer, BertForSequenceClassification
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        
        # 根据模型类型加载模型
        if self.model_type == "onnx":
            onnx_path = os.path.join(self.model_path, "model.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")
            # 动态导入onnxruntime
            import onnxruntime as ort
            self.model = ort.InferenceSession(onnx_path)
            # 添加运行时检查
            if self.model is not None:
                self.input_name = self.model.get_inputs()[0].name
                self.attention_mask_name = self.model.get_inputs()[1].name
                self.output_name = self.model.get_outputs()[0].name
            
        elif self.model_type in ["safetensors", "pytorch"]:
            # 加载PyTorch模型
            model_config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(model_config_path):
                raise FileNotFoundError(f"模型配置文件不存在: {model_config_path}")
            
            # 动态导入torch
            import torch
            
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                use_safetensors=(self.model_type == "safetensors")
            )
            # 添加运行时检查
            if self.model is not None:
                self.model.eval()
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def predict_single(self, text):
        """
        单个文本预测情绪
        
        Args:
            text (str): 输入文本
            
        Returns:
            int: 预测的标签索引
        """
        # 确保tokenizer已正确初始化
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer未正确初始化")
            
        if self.model_type == "onnx":
            # ONNX模型推理
            # 不指定return_tensors，获取原始Encoding对象
            encoding = self.tokenizer(
                text,
                max_length=Config.MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_attention_mask=True
            )
            
            # 动态导入numpy
            import numpy as np
            # 将Encoding对象转换为numpy数组
            input_ids = np.array([encoding["input_ids"]], dtype=np.int64)
            attention_mask = np.array([encoding["attention_mask"]], dtype=np.float32)
            
            # 确保model已正确初始化
            if self.model is None:
                raise RuntimeError("ONNX模型未正确初始化")
                
            # 执行ONNX推理
            outputs = self.model.run(  # type: ignore
                [self.output_name],
                {
                    self.input_name: input_ids,
                    self.attention_mask_name: attention_mask
                }
            )
            
            # outputs是列表，第一个元素是logits
            logits = outputs[0][0]  # type: ignore # shape: (num_labels,)
            prediction = np.argmax(logits)
            return int(prediction)
            
        else:
            # PyTorch模型推理
            inputs = self.tokenizer(
                text,
                max_length=Config.MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 确保model已正确初始化
            if self.model is None:
                raise RuntimeError("PyTorch模型未正确初始化")
            
            # 动态导入torch
            import torch
            with torch.no_grad():
                outputs = self.model(**inputs)  # type: ignore
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1)
                return int(prediction.cpu().numpy()[0])
    
    def evaluate(self, test_csv_path):
        """
        评估模型性能
        
        Args:
            test_csv_path (str): 测试数据集CSV文件路径
            
        Returns:
            dict: 评估结果
        """
        # 动态导入pandas
        import pandas as pd
        
        # 加载测试数据
        df = pd.read_csv(test_csv_path)
        texts = df['text'].tolist()
        true_labels = df['label'].tolist()
        
        # 转换真实标签为索引
        true_label_indices = [self.label2id[label] for label in true_labels]
        
        # 逐个预测
        print("正在执行逐个预测...")
        pred_label_indices = []
        total_samples = len(texts)
        for i, text in enumerate(texts):
            pred = self.predict_single(text)
            pred_label_indices.append(pred)
            if (i + 1) % 50 == 0 or i == total_samples - 1:
                print(f"已处理 {i + 1}/{total_samples} 个样本")
        
        # 动态导入sklearn.metrics
        from sklearn.metrics import classification_report, accuracy_score
        
        # 计算评估指标
        accuracy = accuracy_score(true_label_indices, pred_label_indices)
        report = classification_report(
            true_label_indices, 
            pred_label_indices,
            target_names=[self.id2label[i] for i in range(len(self.id2label))],
            output_dict=True,
            zero_division=0
        )
        
        # 打印详细结果
        print("\n=== 模型评估结果 ===")
        print(f"准确率: {accuracy:.4f}")
        if isinstance(report, dict) and 'weighted avg' in report:
            print(f"加权F1分数: {report['weighted avg']['f1-score']:.4f}")
            print(f"加权精确率: {report['weighted avg']['precision']:.4f}")
            print(f"加权召回率: {report['weighted avg']['recall']:.4f}")
        else:
            print("无法获取加权指标")
        
        print("\n详细分类报告:")
        print(classification_report(
            true_label_indices,
            pred_label_indices,
            target_names=[self.id2label[i] for i in range(len(self.id2label))],
            digits=4,
            zero_division=0
        ))
        
        results = {
            "accuracy": accuracy,
            "classification_report": report
        }
        
        if isinstance(report, dict) and 'weighted avg' in report:
            results.update({
                "f1_weighted": report['weighted avg']['f1-score'],
                "precision_weighted": report['weighted avg']['precision'],
                "recall_weighted": report['weighted avg']['recall']
            })
        
        return results


def evaluate_model(model_path, model_type, test_csv_path):
    """
    评估指定模型
    
    Args:
        model_path (str): 模型路径
        model_type (str): 模型类型 ("onnx", "safetensors", "pytorch")
        test_csv_path (str): 测试数据集路径
    """
    try:
        evaluator = ModelEvaluator(model_path, model_type)
        results = evaluator.evaluate(test_csv_path)
        return results
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        raise