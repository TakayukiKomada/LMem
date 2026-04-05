"""
モデル設定管理
Python基礎: dataclass、型ヒント、デフォルト値
AI語録: hidden_size, num_layers, learning_rate, config
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """モデルのハイパーパラメータ設定"""
    model_name: str = "simple_classifier"
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    vocab_size: int = 32000
    max_length: int = 512
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """学習時の設定"""
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class FullConfig:
    """統合設定"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path) -> None:
        """設定をJSONに保存する"""
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"設定保存: {path}")

    @classmethod
    def load(cls, path: Path) -> "FullConfig":
        """JSONから設定を読み込む"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
        )


def print_config(config: FullConfig) -> None:
    """設定を表示する"""
    print("=== モデル設定 ===")
    for key, value in asdict(config.model).items():
        print(f"  {key}: {value}")
    print("=== 学習設定 ===")
    for key, value in asdict(config.training).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    config = FullConfig()
    print_config(config)
