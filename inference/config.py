
"""
推理配置文件
"""
import os
from train.config import Config


class InferenceConfig:
    """推理配置类"""

    # 模型路径配置
    MODEL_DIR = Config.FINAL_MODEL_DIR  # ONNX模型所在目录

    # 推理参数
    MAX_LENGTH = Config.MAX_LENGTH  # 输入文本最大长度

    # 默认返回的top-k结果数量
    DEFAULT_TOP_K = 1
