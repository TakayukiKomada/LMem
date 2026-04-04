"""
シンプルなニューラルネットワーク定義
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """テキスト分類用のシンプルな全結合ネットワーク"""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """順伝播処理"""
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def create_model(config: dict) -> SimpleClassifier:
    """設定辞書からモデルを生成する"""
    model = SimpleClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_classes=config["num_classes"],
    )
    return model


if __name__ == "__main__":
    config = {
        "input_size": 768,
        "hidden_size": 256,
        "num_classes": 10,
    }
    model = create_model(config)
    x = torch.randn(4, 768)
    output = model(x)
    print(f"出力形状: {output.shape}")
