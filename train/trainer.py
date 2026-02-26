import torch
import numpy as np
import os
import json
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from .config import Config
from .dataset import EmotionDataset

def train_and_evaluate(train_texts, test_texts, train_labels, test_labels, label_encoder):
    """训练和评估18类情绪分类模型"""
    # 1. 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=False)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # 2. 数据编码
    print("\nTokenizing 数据...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH,
    )

    # 3. 创建数据集
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    # 4. 训练配置
    training_args_params = {
        "output_dir": Config.OUTPUT_DIR_BASE,
        "num_train_epochs": Config.NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": Config.PER_DEVICE_EVAL_BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
        "weight_decay": Config.WEIGHT_DECAY,
        "warmup_steps": Config.WARMUP_STEPS,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_weighted",
        "logging_steps": 50,
        "seed": Config.SEED,
        "fp16": torch.cuda.is_available(),
        "report_to": "none"
    }
    
    if hasattr(TrainingArguments, 'neftune_noise_alpha'):
        training_args_params["neftune_noise_alpha"] = Config.NEFTUNE_NOISE_ALPHA
        if Config.NEFTUNE_NOISE_ALPHA > 0.0:
            print(f"已启用NEFTune，噪声强度(alpha): {Config.NEFTUNE_NOISE_ALPHA}")
    else:
        if Config.NEFTUNE_NOISE_ALPHA > 0.0:
            print("警告: 当前transformers版本不支持NEFTune，需要升级到>=4.33.0")
    
    training_args = TrainingArguments(**training_args_params)

    # 5. 自定义评估指标
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        report = classification_report(
            labels, preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        # report 是字典类型，包含accuracy和各类别的统计信息
        # 使用类型转换确保安全访问
        report_dict = dict(report) if isinstance(report, dict) else {}
        accuracy = report_dict.get("accuracy", 0.0)
        weighted_avg = report_dict.get("weighted avg", {})
        f1_score = weighted_avg.get("f1-score", 0.0)
        precision = weighted_avg.get("precision", 0.0)
        recall = weighted_avg.get("recall", 0.0)
        
        return {
            "accuracy": float(accuracy),
            "f1_weighted": float(f1_score),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
        }

    # 6. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n开始训练...")
    trainer.train()

    # 7. 最终评估
    print("\n=== 测试集最终性能 (使用最佳模型) ===")
    eval_results = trainer.evaluate(test_dataset)
    print(f"评估结果: {eval_results}")

    # 获取详细的分类报告
    print("\n详细分类报告:")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    print(classification_report(
        test_labels,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    ))

    # 8. 保存模型和配置
    os.makedirs(Config.FINAL_MODEL_DIR, exist_ok=True)
    print(f"\n保存最佳模型到 {Config.FINAL_MODEL_DIR}...")
    trainer.save_model(Config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)

    # 保存标签映射
    label_mapping_path = os.path.join(Config.FINAL_MODEL_DIR, "label_mapping.json")
    print(f"保存标签映射到 {label_mapping_path}...")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "id2label": {str(i): label for i, label in enumerate(label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(label_encoder.classes_)}
        }, f, ensure_ascii=False, indent=2)

    # 保存训练配置
    training_config_path = os.path.join(Config.FINAL_MODEL_DIR, "training_config.json")
    print(f"保存训练配置到 {training_config_path}...")
    training_config = {
        "data_config": {
            "DATA_DIR": Config.DATA_DIR,
            "DATA_PATH": Config.DATA_PATH
        },
        "model_config": {
            "MODEL_NAME": Config.MODEL_NAME,
            "MAX_LENGTH": Config.MAX_LENGTH,
            "NUM_LABELS": Config.NUM_LABELS,
            "TARGET_EMOTIONS": Config.TARGET_EMOTIONS
        },
        "training_config": {
            "NUM_TRAIN_EPOCHS": Config.NUM_TRAIN_EPOCHS,
            "PER_DEVICE_TRAIN_BATCH_SIZE": Config.PER_DEVICE_TRAIN_BATCH_SIZE,
            "PER_DEVICE_EVAL_BATCH_SIZE": Config.PER_DEVICE_EVAL_BATCH_SIZE,
            "LEARNING_RATE": Config.LEARNING_RATE,
            "WEIGHT_DECAY": Config.WEIGHT_DECAY,
            "WARMUP_STEPS": Config.WARMUP_STEPS,
            "SEED": Config.SEED
        },
        "output_config": {
            "OUTPUT_DIR_BASE": Config.OUTPUT_DIR_BASE,
            "FINAL_MODEL_DIR": Config.FINAL_MODEL_DIR,
            "LOGGING_DIR": Config.LOGGING_DIR
        }
    }
    with open(training_config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)

    print(f"\n模型和配置已保存到 {Config.FINAL_MODEL_DIR}")

    from .onnx_exporter import convert_to_onnx
    convert_to_onnx(Config.FINAL_MODEL_DIR, os.path.join(Config.FINAL_MODEL_DIR, "model.onnx"), max_length=Config.MAX_LENGTH)