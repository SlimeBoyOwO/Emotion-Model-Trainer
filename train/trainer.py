
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
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR_BASE,
        num_train_epochs=Config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=Config.LOGGING_DIR,
        logging_steps=50,
        seed=Config.SEED,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

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
        return {
            "accuracy": report["accuracy"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
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
        digits=4
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

    print(f"\n模型和配置已保存到 {Config.FINAL_MODEL_DIR}")

    from .onnx_exporter import convert_to_onnx
    convert_to_onnx(Config.FINAL_MODEL_DIR, Config.FINAL_MODEL_DIR + "model.onnx", max_length=Config.MAX_LENGTH)
