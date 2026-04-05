"""
ECサイト: ショッピングカートシステム
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Product:
    """商品モデル"""
    product_id: int
    name: str
    price: float
    stock: int
    category_id: int
    is_available: bool = True
    tax_rate: float = 0.10
    discount_rate: float = 0.0
    thumbnail_url: str = ""
    created_at: Optional[datetime] = None


@dataclass
class CartItem:
    """カートアイテム"""
    product_id: int
    name: str
    price: float
    quantity: int
    discount_rate: float = 0.0

    @property
    def subtotal(self) -> float:
        """小計を計算する"""
        discounted = self.price * (1 - self.discount_rate)
        return round(discounted * self.quantity, 2)


class ShoppingCart:
    """ショッピングカート管理"""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.items: list[CartItem] = []
        self.coupon_code: Optional[str] = None
        self.coupon_discount: float = 0.0

    def add_item(self, product: Product, quantity: int = 1) -> None:
        """商品をカートに追加する"""
        if not product.is_available:
            raise ValueError(f"商品 {product.name} は現在利用できません")
        if quantity > product.stock:
            raise ValueError(f"在庫不足: 残り {product.stock} 個")

        for item in self.items:
            if item.product_id == product.product_id:
                item.quantity += quantity
                return

        self.items.append(CartItem(
            product_id=product.product_id,
            name=product.name,
            price=product.price,
            quantity=quantity,
            discount_rate=product.discount_rate,
        ))

    def remove_item(self, product_id: int) -> None:
        """商品をカートから削除する"""
        self.items = [
            item for item in self.items
            if item.product_id != product_id
        ]

    def update_quantity(self, product_id: int, quantity: int) -> None:
        """数量を更新する"""
        for item in self.items:
            if item.product_id == product_id:
                if quantity <= 0:
                    self.remove_item(product_id)
                else:
                    item.quantity = quantity
                return
        raise ValueError(f"商品ID {product_id} がカートに見つかりません")

    def clear(self) -> None:
        """カートを空にする"""
        self.items.clear()
        self.coupon_code = None
        self.coupon_discount = 0.0

    @property
    def subtotal(self) -> float:
        """税抜き小計"""
        return sum(item.subtotal for item in self.items)

    @property
    def tax_amount(self) -> float:
        """税額"""
        return round(self.subtotal * 0.10, 2)

    @property
    def total(self) -> float:
        """合計金額"""
        return self.subtotal + self.tax_amount - self.coupon_discount

    def to_dict(self) -> dict:
        """辞書形式に変換する"""
        return {
            "user_id": self.user_id,
            "items": [
                {
                    "product_id": item.product_id,
                    "name": item.name,
                    "price": item.price,
                    "quantity": item.quantity,
                    "subtotal": item.subtotal,
                }
                for item in self.items
            ],
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "coupon_discount": self.coupon_discount,
            "total": self.total,
        }


if __name__ == "__main__":
    product = Product(product_id=1, name="テスト商品", price=1000, stock=10, category_id=1)
    cart = ShoppingCart(user_id=1)
    cart.add_item(product, quantity=3)
    print(f"小計: {cart.subtotal}")
    print(f"合計: {cart.total}")
