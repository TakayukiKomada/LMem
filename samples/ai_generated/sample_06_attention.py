"""
Attentionメカニズムの実装
Python基礎: 数学的演算、テンソル操作
AI語録: attention, softmax, multi-head, scaled dot-product
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """スケーリング付きドットプロダクトアテンション"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """アテンション計算"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """マルチヘッドアテンションの計算"""
        batch_size = hidden_states.size(0)
        q = self.query(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output = self.attention(q, k, v, attention_mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.output(attention_output)


if __name__ == "__main__":
    mha = MultiHeadAttention(hidden_size=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = mha(x)
    print(f"入力形状: {x.shape}")
    print(f"出力形状: {out.shape}")
