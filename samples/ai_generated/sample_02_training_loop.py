"""
学習ループの基本パターン
Python基礎: forループ、with文、条件分岐
AI語録: optimizer, loss, backward, scheduler
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """シンプルなテキストデータセット"""

    def __init__(self, texts: list[str], labels: list[int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """1エポック分の学習を実行する"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """モデルを評価して正答率を返す"""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / max(total_samples, 1)
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
