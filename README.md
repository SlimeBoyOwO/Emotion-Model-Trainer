# 情感分类器训练器

> 一个简单的用来训练情感语句分类模型的python程序

## 如何使用？

1. 在csv模型里输入情感语句和对应的情感标签
2. 使用csvCleaner.py处理csv文件防止字符污染
3. 运行对应的训练程序
4. 按照不同的需求从以下方法中选择进行训练

    - BERT 为基座模型进行全参数微调
  
      运行startTrain.py等待训练完毕，完成后自动生成模型品质报告

    - S-BERT 为基座模型进行全参数微调

      > 需要安装sentence-transformers库

      运行startTrain_sbert.py等待训练完毕，完成后自动生成报告

    - S-BERT 为基座模型，使用 PEFT-Turning 训练

      > 需要安装sentence-transformers, peft, accelerate库

      运行startTrain_PEFT.py等待训练完毕，完成后自动生成报告

## 如何测试？

  全参数微调的模型可使用testModel.py测试

  使用PEFT-Turning的模型可使用testModel_PEFT.py测试

## 额外功能

> 可以用addData.py人工审核添加情绪
