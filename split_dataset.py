#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割脚本
从每种情感类别中按指定比例抽取样本到test.csv，其余写入train.csv
支持分层抽样，确保每个情感类别在训练集和测试集中都有代表性样本
"""

import pandas as pd
import os
import argparse
import sys
from pathlib import Path


def split_dataset_by_emotion(data_path, test_ratio=0.2, output_dir=None, random_state=42):
    """
    按情感类别分割数据集
    
    Args:
        data_path (str): 输入CSV文件路径
        test_ratio (float): 测试集比例，默认0.2（20%）
        output_dir (str): 输出目录，默认为输入文件所在目录
        random_state (int): 随机种子，确保结果可重现
    """
    # 读取数据
    print(f"正在读取数据文件: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"无法读取数据文件 {data_path}: {e}")
    
    # 检查必要的列是否存在
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"CSV文件必须包含'text'和'label'列。当前列: {list(df.columns)}")
    
    # 检查数据是否为空
    if df.empty:
        raise ValueError("数据文件为空")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(data_path)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    train_data = []
    test_data = []
    
    # 按情感类别分组
    emotion_groups = df.groupby('label')
    
    print(f"发现 {len(emotion_groups)} 种情感类别:")
    for emotion, group in emotion_groups:
        total_count = len(group)
        if total_count == 0:
            continue
            
        # 计算测试集样本数，确保至少有1个样本用于测试（如果总数>=2）
        if total_count == 1:
            # 只有一个样本，放入训练集
            train_data.append(group)
            print(f"  {emotion}: {total_count} -> 全部放入训练集 (仅1个样本)")
        else:
            test_count = max(1, int(total_count * test_ratio))
            # 确保测试集不会超过总样本数
            test_count = min(test_count, total_count - 1)
            
            # 随机抽取测试样本
            test_sample = group.sample(n=test_count, random_state=random_state)
            train_sample = group.drop(test_sample.index)
            
            test_data.append(test_sample)
            train_data.append(train_sample)
            
            print(f"  {emotion}: {total_count} -> 训练集: {len(train_sample)}, 测试集: {len(test_sample)}")
    
    # 合并数据
    if train_data:
        train_df = pd.concat(train_data, ignore_index=True)
    else:
        train_df = pd.DataFrame(columns=['text', 'label'])
        
    if test_data:
        test_df = pd.concat(test_data, ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=['text', 'label'])
    
    # 保存文件
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    # 使用标准utf-8编码（无BOM）
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"\n分割完成!")
    print(f"训练集保存到: {train_path} (共 {len(train_df)} 条样本)")
    print(f"测试集保存到: {test_path} (共 {len(test_df)} 条样本)")
    print(f"总样本数: {len(df)}")
    print(f"实际测试集比例: {len(test_df)/len(df):.1%}")


def main():
    parser = argparse.ArgumentParser(
        description='按情感类别分割数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python split_dataset.py --data-path ./data/emotion_data_manual.csv
  python split_dataset.py --data-path ./data/emotion_data_manual.csv --test-ratio 0.3 --output-dir ./data
        """
    )
    parser.add_argument('--data-path', type=str, required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='测试集比例 (默认: 0.2，范围: 0.01-0.99)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (默认: 输入文件所在目录)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.data_path):
        print(f"错误: 文件 {args.data_path} 不存在", file=sys.stderr)
        sys.exit(1)
    
    if args.test_ratio <= 0 or args.test_ratio >= 1:
        print("错误: 测试集比例必须在 (0, 1) 范围内", file=sys.stderr)
        sys.exit(1)
    
    if args.test_ratio < 0.01:
        print("警告: 测试集比例过小 (<1%)，可能导致某些类别没有测试样本", file=sys.stderr)
    
    try:
        split_dataset_by_emotion(
            data_path=args.data_path,
            test_ratio=args.test_ratio,
            output_dir=args.output_dir,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()