
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import Config

def load_data(data_path=None):
    """加载并预处理数据，强制使用定义的情绪标签"""
    if data_path is None:
        data_path = Config.DATA_PATH

    # 加载数据并筛选目标情绪
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误：数据文件 '{data_path}' 未找到。请确保文件存在于正确路径。")
        exit()

    # 筛选有效标签，并转换为字符串以防万一
    data["label"] = data["label"].astype(str)
    data = data[data["label"].isin(Config.TARGET_EMOTIONS)].copy()
    data["text"] = data["text"].astype(str)

    if data.empty:
        print(f"错误：在 '{data_path}' 中没有找到属于 TARGET_EMOTIONS 的数据。")
        exit()

    # 数据统计
    print("\n=== 数据统计 ===")
    print(f"目标情绪类别数量: {Config.NUM_LABELS}")
    print("筛选后总样本数:", len(data))
    print("类别分布:\n", data["label"].value_counts())

    print(len(Config.TARGET_EMOTIONS))
    print(Config.TARGET_EMOTIONS)
    # 等待输入
    input("按回车键继续...")

    # 使用固定顺序的标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(Config.TARGET_EMOTIONS)  # 强制按定义顺序编码

    # 划分数据集（保证测试集至少包含每个类别一个样本，如果可能）
    # 计算最小测试集比例以包含所有类
    min_samples_per_class = 1
    required_test_samples = Config.NUM_LABELS * min_samples_per_class
    min_test_size_for_all_classes = required_test_samples / len(data)

    # 设置测试集比例，通常在0.1到0.3之间，但要确保能覆盖所有类
    test_size = max(0.2, min(min_test_size_for_all_classes, 0.3))
    # 如果总样本太少，可能无法满足 stratify 要求，这里简化处理
    if len(data) < Config.NUM_LABELS * 2: # 至少保证训练集和测试集每个类都有样本（理论上）
         print("警告：数据量过少，可能无法有效分层或训练。")
         test_size = max(0.1, min_test_size_for_all_classes) # 尝试减少测试集比例

    print(f"实际使用的测试集比例: {test_size:.2f}")

    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data["text"].tolist(),
            data["label"].tolist(),
            test_size=test_size,
            stratify=data["label"], # 尝试分层抽样
            random_state=Config.SEED
        )
    except ValueError as e:
        print(f"分层抽样失败: {e}. 可能某些类别样本过少。尝试非分层抽样...")
        # 如果分层失败（通常因为某类样本太少），退回到普通随机抽样
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data["text"].tolist(),
            data["label"].tolist(),
            test_size=test_size,
            random_state=Config.SEED
        )

    # 编码标签
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    print(f"\n划分结果: 训练集={len(train_texts)}, 测试集={len(test_texts)}")
    print("测试集类别分布:\n", pd.Series(test_labels).value_counts().sort_index())
    # 检查测试集是否包含所有类别
    test_unique_labels = set(test_labels)
    if len(test_unique_labels) < Config.NUM_LABELS:
        print(f"警告：测试集仅包含 {len(test_unique_labels)}/{Config.NUM_LABELS} 个类别。缺失的类别：{set(Config.TARGET_EMOTIONS) - test_unique_labels}")

    print("标签映射:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    return train_texts, test_texts, train_labels_encoded, test_labels_encoded, label_encoder
