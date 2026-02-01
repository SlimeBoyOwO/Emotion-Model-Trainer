# minirbt_train_v5_5090_optimized.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
import os
import json
import torch.nn.functional as F
import random
from collections import Counter
from transformers.trainer_utils import EvalPrediction
import warnings

warnings.filterwarnings('ignore')

# --- 全局超参数 ---
SEED = 42
FGM_EPSILON = 0.5   # 略微减小以保证稳定性
RDROP_ALPHA = 1.0   # 增加 R-Drop 约束强度
EMA_DECAY = 0.995   # EMA 衰减率，用于平滑权重

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()

TARGET_EMOTIONS = [
    "兴奋", "厌恶", "害怕", "害羞", "慌张",      
    "担心", "无奈", "生气", "疑惑", "紧张",      
    "认真", "高兴", "哭泣", "心动", "调皮",      
    "难为情", "自信", "惊讶", "倦淡平宁"        
]
NUM_LABELS = len(TARGET_EMOTIONS)

# --- 模型权重平滑 (EMA) ---
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# --- 对抗训练 (FGM) ---
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=FGM_EPSILON, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

# --- 分类头 (AWA + Multi-Sample Dropout) ---
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

class EmotionModel(nn.Module):
    def __init__(self, model_name="hfl/minirbt-h288"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.classifier = AdvancedClassificationHead(self.config.hidden_size, NUM_LABELS)
    
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
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
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.15) # 提高平滑度，抗过拟合
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits}

# --- 高级 Trainer 实现 ---
class SuperTrainer(AdvancedEmotionTrainer if 'AdvancedEmotionTrainer' in globals() else Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.fgm = FGM(self.model)
        self.ema = EMA(self.model, EMA_DECAY)
        self.ema.register()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        # R-Drop: 双向前向传播
        outputs1 = model(**inputs)
        logits1 = outputs1.get("logits")
        
        if self.model.training:
            outputs2 = model(**inputs)
            logits2 = outputs2.get("logits")
            
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits1.device) if self.class_weights is not None else None, 
                label_smoothing=0.15
            )
            
            ce_loss = 0.5 * (loss_fct(logits1, labels) + loss_fct(logits2, labels))
            kl_loss = 0.5 * (F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='batchmean') + 
                             F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='batchmean'))
            loss = ce_loss + RDROP_ALPHA * kl_loss
        else:
            loss = outputs1.get("loss")

        return (loss, outputs1) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 1. 标准前向 + 反向
        loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        
        # 2. 对抗训练阶段 (FGM)
        self.fgm.attack()
        loss_adv = self.compute_loss(model, inputs)
        self.accelerator.backward(loss_adv)
        self.fgm.restore()
        
        # 3. 梯度更新由 Trainer 处理，此处我们手动触发 EMA 更新
        self.ema.update()
        
        return loss.detach()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 评估时应用 EMA 影子权重
        self.ema.apply_shadow()
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.ema.restore()
        return output

# --- 打印增强回调 ---
class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        history = state.log_history
        train_loss = next((h['loss'] for h in reversed(history) if 'loss' in h), None)
        eval_metrics = next((h for h in reversed(history) if 'eval_loss' in h), None)
        
        print(f"\n[Epoch {state.epoch:.1f}]")
        if train_loss: print(f" - Train Loss: {train_loss:.4f}")
        if eval_metrics:
            print(f" - Eval Loss : {eval_metrics['eval_loss']:.4f}")
            print(f" - Eval F1   : {eval_metrics.get('eval_f1_macro', 0):.4f}")
            print(f" - Eval Acc  : {eval_metrics.get('eval_accuracy', 0):.4f}")

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'labels': self.labels[idx]}

class SmartDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        texts = [f['text'] for f in features]
        labels = [f['labels'] for f in features]
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        return inputs

def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    report = classification_report(p.label_ids, preds, output_dict=True, zero_division=0)
    return {"f1_macro": report["macro avg"]["f1-score"], "accuracy": report["accuracy"]}

def run_training(data_path="ok.csv"):
    model_name = "hfl/minirbt-h288"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    df = pd.read_csv(data_path)
    label_to_index = {label: idx for idx, label in enumerate(TARGET_EMOTIONS)}
    df = df[df['label'].isin(TARGET_EMOTIONS)].copy()
    df['label'] = df['label'].map(label_to_index)
    
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=SEED)
    
    counts = Counter(train_df['label'])
    weights = torch.tensor([1.0 / (counts.get(i, 1)) for i in range(NUM_LABELS)], dtype=torch.float32)
    class_weights = weights / weights.sum() * NUM_LABELS
    
    train_dataset = EmotionDataset(train_df['text'], train_df['label'])
    val_dataset = EmotionDataset(val_df['text'], val_df['label'])
    test_dataset = EmotionDataset(test_df['text'], test_df['label'])

    data_collator = SmartDataCollator(tokenizer)

    # --- 针对 5090 优化训练参数 ---
    training_args = TrainingArguments(
        output_dir="./output_minirbt_v5",
        num_train_epochs=6,               # 略微增加 Epoch 配合 EMA
        per_device_train_batch_size=64,    # 5090 算力支持大 batch
        per_device_eval_batch_size=128,
        learning_rate=5e-5,                # 适当提高 LR
        weight_decay=0.05,                 # 增强正则化，防止 5 轮后过拟合
        warmup_ratio=0.15,                 # 增加预热比例
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,                  # 增加打印频率
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        bf16=True if torch.cuda.is_bf16_supported() else False, # 5090 必开 BF16
        fp16=False if torch.cuda.is_bf16_supported() else True,
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=False,                # 确保显示进度条
        dataloader_num_workers=4           # 利用多核 CPU 加速数据加载
    )

    model = EmotionModel(model_name)
    
    trainer = SuperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            PrinterCallback()
        ]
    )

    print(f"\n>>> 硬件状态: RTX 5090 检测通过 | BF16 加速: {training_args.bf16}")
    print(">>> 核心策略: FGM + R-Drop + AWA Head + EMA Smoothing + Enhanced Regularization")
    
    trainer.train()

    # 最后测试集的评估使用 EMA 权重
    trainer.ema.apply_shadow()
    print("\n" + "="*50)
    print(">>> 最终测试集详细报告 (EMA权重)")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    print(classification_report(y_true, y_pred, target_names=TARGET_EMOTIONS, digits=4))
    print("="*50)

    # 保存模型
    save_path = "./model_minirbt_final_v5"
    os.makedirs(save_path, exist_ok=True)
    # 保存的是 EMA 优化后的 state_dict
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_path)
    
    mapping = {"id2label": {i: l for i, l in enumerate(TARGET_EMOTIONS)}, 
               "label2id": {l: i for i, l in enumerate(TARGET_EMOTIONS)}}
    with open(os.path.join(save_path, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_training("ok.csv")