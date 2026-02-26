import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置TensorBoard日志目录环境变量
os.environ["TENSORBOARD_LOGGING_DIR"] = os.getenv("LOGGING_DIR", "./logs_19emo")

class Config:
    """训练配置类，从环境变量加载配置"""

    # 数据配置
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    DATA_PATH = os.getenv("DATA_PATH", "./data/emotion_data_manual.csv")

    # 模型配置
    MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-chinese")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
    NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 8))
    PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 16))
    PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", 32))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    # WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", 0.1))  # 已弃用
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 500))  # 使用warmup_steps替代
    SEED = int(os.getenv("SEED", 42))

    # NEFTune配置
    NEFTUNE_NOISE_ALPHA = float(os.getenv("NEFTUNE_NOISE_ALPHA", 0.0))  # 0.0表示禁用NEFTune

    # 输出路径
    OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "./results_19emo")
    FINAL_MODEL_DIR = os.getenv("FINAL_MODEL_DIR", "./emotion_model_19emo")
    LOGGING_DIR = os.getenv("LOGGING_DIR", "./logs_19emo")

    # 19类情绪标签
    TARGET_EMOTIONS = os.getenv("TARGET_EMOTIONS", 
                               "高兴,平静,厌恶,害羞,害怕,生气,认真,紧张,慌张,疑惑,兴奋,无奈,担心,惊讶,哭泣,心动,难为情,自信,调皮").split(",")
    NUM_LABELS = len(TARGET_EMOTIONS)