"""
金融: ポートフォリオ分析
Python基礎: 数値計算、統計処理、辞書内包表記
金融語録: portfolio, return, sharpe_ratio, volatility
"""
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StockPrice:
    """株価データ"""
    ticker: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: float


@dataclass
class Position:
    """ポジション"""
    ticker: str
    quantity: int
    execution_price: float
    current_price: float = 0.0
    commission: float = 0.0

    @property
    def position_size(self) -> float:
        """ポジションサイズ"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """含み損益"""
        return (self.current_price - self.execution_price) * self.quantity - self.commission

    @property
    def total_return(self) -> float:
        """リターン率"""
        cost = self.execution_price * self.quantity + self.commission
        if cost == 0:
            return 0.0
        return self.unrealized_pnl / cost * 100


class Portfolio:
    """ポートフォリオ管理"""

    def __init__(self):
        self.positions: list[Position] = []
        self.cash_balance: float = 0.0
        self.realized_pnl: float = 0.0

    @property
    def portfolio_value(self) -> float:
        """ポートフォリオ総額"""
        return sum(p.position_size for p in self.positions) + self.cash_balance

    def buy_order(self, ticker: str, quantity: int, price: float, commission: float = 0.0) -> None:
        """買い注文を実行する"""
        total_cost = price * quantity + commission
        if total_cost > self.cash_balance:
            raise ValueError(f"残高不足: 必要 {total_cost:.2f}, 残高 {self.cash_balance:.2f}")

        self.cash_balance -= total_cost

        for pos in self.positions:
            if pos.ticker == ticker:
                total_qty = pos.quantity + quantity
                pos.execution_price = (pos.execution_price * pos.quantity + price * quantity) / total_qty
                pos.quantity = total_qty
                pos.commission += commission
                return

        self.positions.append(Position(
            ticker=ticker,
            quantity=quantity,
            execution_price=price,
            current_price=price,
            commission=commission,
        ))

    def sell_order(self, ticker: str, quantity: int, price: float, commission: float = 0.0) -> float:
        """売り注文を実行する"""
        for pos in self.positions:
            if pos.ticker == ticker:
                if quantity > pos.quantity:
                    raise ValueError(f"保有数量不足: {pos.quantity} 株")
                pnl = (price - pos.execution_price) * quantity - commission
                self.realized_pnl += pnl
                self.cash_balance += price * quantity - commission
                pos.quantity -= quantity
                if pos.quantity == 0:
                    self.positions.remove(pos)
                return pnl
        raise ValueError(f"ポジションなし: {ticker}")

    def update_prices(self, prices: dict[str, float]) -> None:
        """現在価格を更新する"""
        for pos in self.positions:
            if pos.ticker in prices:
                pos.current_price = prices[pos.ticker]


def calculate_returns(prices: list[float]) -> list[float]:
    """価格系列からリターンを計算する"""
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]


def calculate_volatility(returns: list[float]) -> float:
    """ボラティリティ（標準偏差）を計算する"""
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.02,
    annualize: bool = True,
) -> float:
    """シャープレシオを計算する"""
    if not returns:
        return 0.0
    mean_return = sum(returns) / len(returns)
    volatility = calculate_volatility(returns)
    if volatility == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    sharpe = (mean_return - daily_rf) / volatility
    if annualize:
        sharpe *= math.sqrt(252)
    return round(sharpe, 4)


def calculate_max_drawdown(prices: list[float]) -> float:
    """最大ドローダウンを計算する"""
    if not prices:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 4)


if __name__ == "__main__":
    pf = Portfolio()
    pf.cash_balance = 1000000
    pf.buy_order("AAPL", 10, 150.0, commission=10)
    pf.buy_order("GOOG", 5, 2800.0, commission=10)
    pf.update_prices({"AAPL": 160.0, "GOOG": 2900.0})
    print(f"ポートフォリオ総額: {pf.portfolio_value:,.2f}")
    for pos in pf.positions:
        print(f"  {pos.ticker}: {pos.unrealized_pnl:+,.2f} ({pos.total_return:+.2f}%)")
