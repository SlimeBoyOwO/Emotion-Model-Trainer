import re
import pandas as pd
from unicodedata import normalize

def deep_clean_text(text):
    """
    深度清洗文本的核武器函数
    处理：不可见字符、异常空格、BOM头、零宽字符、HTML标签等
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 标准化Unicode字符（如全角转半角）
    text = normalize('NFKC', text)
    
    # 2. 移除不可打印字符（保留中文、英文、数字、常用标点）
    text = ''.join(
        char for char in text 
        if char.isprintable() or 
           '\u4e00' <= char <= '\u9fff'  # 保留所有汉字
    )
    
    # 3. 处理特殊空白字符
    text = re.sub(r'[\u200b-\u200d\u2028-\u202f\ufeff]', '', text)  # 零宽字符
    
    # 4. 替换异常空格（包含全角空格）
    text = re.sub(r'[\s\u3000]+', ' ', text).strip()
    
    # 5. 移除控制字符（ASCII 0-31）
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    
    # 6. 处理HTML/XML标签（如果有）
    text = re.sub(r'<[^>]+>', '', text)
    
    # 7. 处理URL和特殊符号
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[【】■◆▶☞➤]+', '', text)  # 去除装饰性符号
    
    return text

def clean_csv_file(input_path, output_path):
    """
    清洗整个CSV文件
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    """
    # 读取数据（强制UTF-8编码，处理BOM头）
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='gbk')
    
    # 检查必要列是否存在
    assert 'text' in df.columns, "CSV文件必须包含'text'列"
    
    # 深度清洗
    print("正在清洗数据...")
    df['text'] = df['text'].astype(str).apply(deep_clean_text)
    
    # 移除空文本
    df = df[df['text'].str.strip().astype(bool)]
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"清洗完成！已保存到 {output_path}")
    
    # 打印清洗报告
    sample_report = df.sample(min(5, len(df)))
    print("\n清洗后样本示例:")
    print(sample_report[['text']].to_markdown(index=False))

if __name__ == "__main__":
    input_file = "emotion_data_manual.csv"  # 原始数据路径
    output_file = "emotion_data_manual.csv"  # 清洗后保存路径
    
    clean_csv_file(input_file, output_file)
    
    # 验证清洗效果
    print("\n验证清洗结果:")
    test_samples = [
        "  Hello\u200bWorld！  ",  # 含零宽空格
        "NUL\x00字符测试",         # 含控制字符
        "全角　空格\u3000测试",     # 全角空格
        "<p>HTML标签</p>",         # HTML标签
        "https://example.com 网址"  # URL
    ]
    
    print("\n清洗测试案例:")
    for sample in test_samples:
        cleaned = deep_clean_text(sample)
        print(f"原始: {repr(sample)} → 清洗后: {repr(cleaned)}")