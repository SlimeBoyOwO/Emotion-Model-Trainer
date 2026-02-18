
import torch

class EmotionDataset(torch.utils.data.Dataset):
    """自定义数据集类"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # 确保编码包含张量或转换numpy数组
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
