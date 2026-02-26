# 情绪分类模型训练项目

本项目用于训练一个能够识别19+种不同情绪的文本分类模型。

## 项目结构

```
New-Emotion-Trainer/
├── data/                    # 数据目录
│   └── emotion_data_manual.csv
├── pre-process/             # 数据预处理
│   └── csvCleaner.py
├── train/                   # 训练模块
│   ├── __init__.py
│   ├── config.py           # 配置模块
│   ├── data_loader.py      # 数据加载模块
│   ├── dataset.py          # 数据集类
│   └── trainer.py          # 训练模块
├── inference/              # 推理模块
│   ├── __init__.py
│   ├── config.py           # 推理设置模块
│   └── emotion_predictor.py# 推理类模块
├── .env                    # 环境变量配置
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

项目使用.env文件进行配置，包含以下主要配置项：

- 数据路径
- 模型参数
- 训练超参数
- 输出路径
- 情绪标签列表
- **NEFTune噪声强度** (NEFTUNE_NOISE_ALPHA): 设置为0.0表示禁用NEFTune

## 使用方法

运行训练程序：

```bash
python main.py
```

## 情绪类别

模型可以识别以下19种情绪：
可以在env和data中的csv新增识别情绪

1. 高兴
2. 厌恶
3. 害羞
4. 害怕
5. 生气
6. 认真
7. 紧张
8. 慌张
9. 疑惑
10. 兴奋
11. 无奈
12. 担心
13. 惊讶
14. 哭泣
15. 心动
16. 难为情
17. 自信
18. 调皮
19. 平静

## NEFTune支持

本项目支持NEFTune技术，通过在嵌入层添加噪声来提高模型性能。要启用NEFTune，请在`.env`文件中设置`NEFTUNE_NOISE_ALPHA`为正值（如0.1）。注意：需要transformers>=4.33.0版本。