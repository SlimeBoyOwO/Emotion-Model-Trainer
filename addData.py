# emotion_data_entry.py
import pandas as pd
import os


def add_to_csv():
    """交互式添加情绪数据到CSV文件"""
    # 检查文件是否存在
    file_path = "emotion_data_manual.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"\n当前已有数据: {len(df)}条")
    else:
        df = pd.DataFrame(columns=["text", "label"])
        print("新建数据文件...")

    # 可用情绪列表
    emotions = [
        "高兴",
        "厌恶",
        "害羞",
        "害怕",
        "生气",
        "认真",
        "紧张",
        "慌张",
        "疑惑",
        "兴奋",
        "无奈",
        "担心",
        "惊讶",
        "哭泣",
        "心动",
        "难为情",
        "自信",
        "调皮",
    ]

    while True:
        print("\n" + "=" * 30)
        print("当前可用情绪标签:")
        print(", ".join(emotions))
        print("=" * 30)

        # 输入文本
        text = input("请输入文本内容（输入q退出）: ").strip()
        if text.lower() == "q":
            break

        # 输入标签
        while True:
            label = input("请输入情绪标签: ").strip()
            if label in emotions:
                break
            print(f"无效标签！请从以下选择: {', '.join(emotions)}")

        # 添加到DataFrame
        new_row = {"text": text, "label": label}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # 保存到CSV
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"已添加: 「{text}」→ {label} (当前总数: {len(df)})")


if __name__ == "__main__":
    print("=== 情绪数据录入工具 ===")
    print("说明: 逐条添加文本和对应情绪标签")
    add_to_csv()
    print("\n数据已保存到 emotion_data_manual.csv")
