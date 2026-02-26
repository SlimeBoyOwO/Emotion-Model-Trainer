import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import Config
import os

def load_data(data_path=None):
    """加载并预处理数据，强制使用定义的情绪标签"""
    if data_path is None:
        data_path = Config.DATA_PATH

    # 检查是否存在现成的train.csv和test.csv文件
    data_dir = os.path.dirname(data_path) if data_path else Config.DATA_DIR
    train_csv_path = os.path.join(data_dir, 'train.csv')
    test_csv_path = os.path.join(data_dir, 'test.csv')
    
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        print(f"检测到现成的训练集和测试集文件:")
        print(f"  训练集: {train_csv_path}")
        print(f"  测试集: {test_csv_path}")
        print("将直接加载现成的数据集，跳过自动分割步骤。")
        
        # 加载训练集
        try:
            train_data = pd.read_csv(train_csv_path)
        except FileNotFoundError:
            print(f"错误：训练集文件 '{train_csv_path}' 未找到。")
            exit()
            
        # 加载测试集
        try:
            test_data = pd.read_csv(test_csv_path)
        except FileNotFoundError:
            print(f"错误：测试集文件 '{test_csv_path}' 未找到。")
            exit()
        
        # 筛选有效标签，并转换为字符串以防万一
        train_data["label"] = train_data["label"].astype(str)
        train_data = train_data[train_data["label"].isin(Config.TARGET_EMOTIONS)].copy()
        train_data["text"] = train_data["text"].astype(str)
        
        test_data["label"] = test_data["label"].astype(str)
        test_data = test_data[test_data["label"].isin(Config.TARGET_EMOTIONS)].copy()
        test_data["text"] = test_data["text"].astype(str)

        if train_data.empty or test_data.empty:
            print(f"错误：在现成的数据集中没有找到属于 TARGET_EMOTIONS 的数据。")
            exit()

        # 数据统计
        print("\n=== 数据统计 ===")
        print(f"目标情绪类别数量: {Config.NUM_LABELS}")
        print(f"训练集样本数: {len(train_data)}")
        print(f"测试集样本数: {len(test_data)}")
        print("训练集类别分布:\n", train_data["label"].value_counts())
        print("测试集类别分布:\n", test_data["label"].value_counts())

        # 使用固定顺序的标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(Config.TARGET_EMOTIONS)  # 强制按定义顺序编码

        # 提取文本和标签
        train_texts = train_data["text"].tolist()
        train_labels = train_data["label"].tolist()
        test_texts = test_data["text"].tolist()
        test_labels = test_data["label"].tolist()

        # 编码标签
        train_labels_encoded = label_encoder.transform(train_labels)
        test_labels_encoded = label_encoder.transform(test_labels)

        print(f"\n最终数据集大小: 训练集={len(train_texts)}, 测试集={len(test_texts)}")
        # 创建标签映射字典
        label_mapping = {label: idx for idx, label in enumerate(Config.TARGET_EMOTIONS)}
        print("标签映射:", label_mapping)

        return train_texts, test_texts, train_labels_encoded, test_labels_encoded, label_encoder
    
    else:
        print("未检测到现成的train.csv和test.csv文件，将从原始数据进行自动分割...")
        # 原有的自动分割逻辑
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

        # 创建标签映射字典
        label_mapping = {label: idx for idx, label in enumerate(Config.TARGET_EMOTIONS)}
        print("标签映射:", label_mapping)

        return train_texts, test_texts, train_labels_encoded, test_labels_encoded, label_encoder