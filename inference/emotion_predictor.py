
import os
import json
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer


class EmotionPredictor:
    """使用ONNX模型进行情绪预测的推理器"""

    def __init__(self, model_dir, max_length=128):
        """
        初始化情绪预测器

        Args:
            model_dir (str): 模型目录路径，包含model.onnx和label_mapping.json
            max_length (int): 输入文本的最大长度
        """
        self.model_dir = model_dir
        self.max_length = max_length

        # 加载ONNX模型
        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")

        self.session = ort.InferenceSession(onnx_path)

        # 加载标签映射
        label_mapping_path = os.path.join(model_dir, "label_mapping.json")
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在: {label_mapping_path}")

        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
            self.id2label = label_mapping["id2label"]
            self.label2id = label_mapping["label2id"]

        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.attention_mask_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

        # 打印模型输入输出信息
        print(f"[EmotionPredictor] 模型输入:")
        for input in self.session.get_inputs():
            print(f"  名称: {input.name}, 类型: {input.type}, 形状: {input.shape}")
        print(f"[EmotionPredictor] 模型输出:")
        for output in self.session.get_outputs():
            print(f"  名称: {output.name}, 类型: {output.type}, 形状: {output.shape}")

        print(f"[EmotionPredictor] 模型加载成功: {onnx_path}")
        print(f"[EmotionPredictor] 支持的情绪类别: {len(self.id2label)}")

    def predict(self, text, return_top_k=1):
        """
        预测文本的情绪

        Args:
            text (str): 输入文本
            return_top_k (int): 返回前k个最可能的情绪类别

        Returns:
            如果return_top_k=1: 返回预测的情绪标签
            如果return_top_k>1: 返回前k个情绪标签及其概率的列表，格式为[(label, probability), ...]
        """
        # 文本预处理
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        # 转换输入数据类型：input_ids保持int64，attention_mask转换为float32
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.float32)

        # 获取模型输出
        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_ids,
                self.attention_mask_name: attention_mask
            }
        )

        # 计算概率
        logits = outputs[0][0]  # shape: (num_labels,)
        probabilities = self._softmax(logits)

        # 获取top-k结果
        top_k_indices = np.argsort(probabilities)[::-1][:return_top_k]

        if return_top_k == 1:
            return self.id2label[str(top_k_indices[0])]
        else:
            return [
                (self.id2label[str(idx)], float(probabilities[idx]))
                for idx in top_k_indices
            ]

    def predict_batch(self, texts, return_top_k=1):
        """
        批量预测文本的情绪

        Args:
            texts (list[str]): 输入文本列表
            return_top_k (int): 返回前k个最可能的情绪类别

        Returns:
            如果return_top_k=1: 返回预测的情绪标签列表
            如果return_top_k>1: 返回前k个情绪标签及其概率的列表的列表
        """
        # 文本预处理
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        # 转换输入数据类型：input_ids保持int64，attention_mask转换为float32
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.float32)

        # 获取模型输出
        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_ids,
                self.attention_mask_name: attention_mask
            }
        )

        # 计算概率
        logits = outputs[0]  # shape: (batch_size, num_labels)
        probabilities = self._softmax(logits, axis=1)

        # 获取top-k结果
        top_k_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :return_top_k]

        if return_top_k == 1:
            return [self.id2label[str(idx[0])] for idx in top_k_indices]
        else:
            return [
                [
                    (self.id2label[str(idx)], float(probabilities[i][idx]))
                    for idx in top_k_indices[i]
                ]
                for i in range(len(texts))
            ]

    @staticmethod
    def _softmax(x, axis=None):
        """计算softmax概率"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
