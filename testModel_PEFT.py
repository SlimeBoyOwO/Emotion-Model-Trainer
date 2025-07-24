# testModel_PEFT.py
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from peft import PeftModel


class EmotionAndEmbeddingModel:
    def __init__(self, path="./results_18emo_output/emotion_model_18emo"):
        """加载PEFT-Turned S-BERT模型"""
        base_model_path = os.path.join(path, "base_model")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        label_mapping_path = os.path.join(path, "label_mapping.json")
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_config = json.load(f)
        self.id2label = label_config["id2label"]
        self.label2id = label_config["label2id"]

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=18,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,  # 阻止自动初始化分类头
        )  # 初始化分类任务
        self.model = PeftModel.from_pretrained(base_model, path)  # 将adapter和分类头覆盖到模型上
        self.model.to(self.device).eval()

        print(self.model)

        print("=" * 20)
        print("\n加载的标签映射关系:")
        for id, label in self.id2label.items():
            print(f"{id}: {label}")

    def predict_emotion(self, text, confidence_threshold=0.2) -> dict:
        """带置信度过滤的情绪预测"""
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        pred_prob, pred_id = torch.max(probs, dim=1)
        pred_prob = pred_prob.item()
        pred_id = pred_id.item()

        top3 = self._get_top3(probs)

        if pred_prob < confidence_threshold:
            return {
                "label": "不确定",
                "confidence": pred_prob,
                "top3": top3,
                "warning": f"置信度低于阈值({confidence_threshold:.0%})",
            }

        return {"label": self.id2label[str(pred_id)], "confidence": pred_prob, "top3": top3}

    def get_embedding(self, sentences, normalize=True):
        """获取句子向量，当前为BAAI/bge-base-zh-v1.5特化版"""

        if isinstance(sentences, str):
            # bge-base-zh-v1.5为检索任务做了特化，在句子开头添加 "为这个句子生成表示以用于检索相关文章：" 模型效果会更好
            # sentences = "为这个句子生成表示以用于检索相关文章：" + sentences
            sentences = [sentences]

        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
            self.device
        )
        base_model = self.model.get_base_model()  # 获得原始分类任务模型
        encoder = getattr(base_model, base_model.config.model_type)  # 获得原始基座任务模型

        with torch.no_grad():
            # 直接调用基座模型的编码器
            outputs = encoder(**inputs)

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # bge-base-chinese-v1.5使用 CLS Pooling
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0]
        if normalize:
            cls_embedding = F.normalize(cls_embedding, p=2, dim=1)

        # text2vec-base-chinese采用 mean pooling
        # cls_embedding = mean_pooling(outputs, inputs['attention_mask'])

        return cls_embedding.cpu().numpy()

    def _get_top3(self, probs) -> list:
        top3_probs, top3_ids = torch.topk(probs, 3)
        return [
            {"label": self.id2label[str(idx.item())], "probability": prob.item()}
            for prob, idx in zip(top3_probs[0], top3_ids[0])
        ]


def handle_embedding_comparison(model_handler):
    print("\n进入句子相似度比较模式")
    try:
        source_sentence = input("请输入源句子: ").strip()
        if not source_sentence:
            print("源句子不能为空！")
            return

        target_sentences = []
        print("请输入目标句子，每行一个。当不输入任何内容并回车时结束:")
        while True:
            line = input().strip()
            if not line:
                break
            target_sentences.append(line)

        if not target_sentences:
            print("没有输入任何目标句子，操作取消。")
            return

        print("\n正在计算相似度...")

        all_sentences = [source_sentence] + target_sentences
        all_embeddings = model_handler.get_embedding(all_sentences)

        source_embedding = all_embeddings[0]
        target_embeddings = all_embeddings[1:]

        # (N, D) * (D,) -> (N,)  N目标句子数, D向量维度
        similarities = np.dot(target_embeddings, source_embedding)

        results = sorted(zip(target_sentences, similarities), key=lambda item: item[1], reverse=True)

        print("\n" + "=" * 40)
        print(f"源句子: '{source_sentence}'")
        print("相似度排序结果:")
        print("-" * 20)
        for i, (sentence, score) in enumerate(results):
            print(f"{i+1}. 得分: {score:.4f} | 句子: {sentence}")
        print("=" * 40)

    except Exception as e:
        print(f"\n❌ 在比较过程中发生错误: {e}")


def main():
    print("【情绪分析 & 句向量生成器】")
    print("=" * 40)

    try:
        model_handler = EmotionAndEmbeddingModel()
        print("\n模型加载成功！")
        print("输入文本进行分析。")
        print("输入 ':q' 退出。")
        print("输入 ':embed <文本>' 获取句向量。")
        print("输入 ':compare' 或 ':sim' 进入相似度比较模式。")
        print("=" * 40)
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        print("请检查模型路径和文件是否完整。")
        return

    while True:
        try:
            text_input = input("\n请输入指令或文本: ").strip()

            if text_input.lower() in [":q", ":quit", "exit"]:
                print("\n退出程序")
                break

            if not text_input:
                print("输入不能为空！")
                continue

            if text_input.lower().startswith(":embed "):
                text_to_embed = text_input[len(":embed ") :].strip()
                if not text_to_embed:
                    print("请输入需要编码的文本内容！")
                    continue
                embedding = model_handler.get_embedding(text_to_embed)
                print(f"向量维度: {embedding.shape}, 预览: {embedding[0][:5]}...")

            elif text_input.lower() in [":compare", ":sim"]:
                handle_embedding_comparison(model_handler)

            else:
                print("\n" + "=" * 30)
                print("🎯 模式: 情感分类")
                print(f"📝 文本: {text_input}")
                result = model_handler.predict_emotion(text_input)

                if "warning" in result:
                    print(f"⚠️ {result['warning']}")
                print(f"主情绪: {result['label']} (置信度: {result['confidence']:.2%})")

                if result["label"] != "不确定":
                    print("\n其他可能情绪:")
                    for i, item in enumerate(result["top3"][1:], 1):
                        print(f"{i}. {item['label']}: {item['probability']:.2%}")
                print("=" * 30)

        except KeyboardInterrupt:
            print("\n检测到中断，退出程序...")
            break
        except Exception as e:
            print(f"\n❌ 预测时发生错误: {e}")


if __name__ == "__main__":
    main()
