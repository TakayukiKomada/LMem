"""
SQLAlchemy データベースモデル
Python基礎: ORM定義、リレーション、CRUD操作
DB語録: Column, ForeignKey, relationship, Session
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import Session, sessionmaker, relationship, declarative_base

Base = declarative_base()


class User(Base):
    """ユーザーテーブル"""
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    orders = relationship("Order", back_populates="user")
    reviews = relationship("Review", back_populates="user")


class Product(Base):
    """商品テーブル"""
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(200), nullable=False)
    product_description = Column(Text)
    unit_price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    category_id = Column(Integer, ForeignKey("categories.category_id"))
    is_available = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    category = relationship("Category", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")
    reviews = relationship("Review", back_populates="product")


class Category(Base):
    """カテゴリテーブル"""
    __tablename__ = "categories"

    category_id = Column(Integer, primary_key=True)
    category_name = Column(String(100), nullable=False)
    sort_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    products = relationship("Product", back_populates="category")


class Order(Base):
    """注文テーブル"""
    __tablename__ = "orders"

    order_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    order_status = Column(String(50), default="pending")
    subtotal = Column(Float, default=0)
    tax_amount = Column(Float, default=0)
    shipping_fee = Column(Float, default=0)
    total_amount = Column(Float, default=0)
    payment_method = Column(String(50))
    payment_status = Column(String(50), default="unpaid")
    shipping_address = Column(Text)
    order_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    """注文明細テーブル"""
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.order_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    line_total = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


class Review(Base):
    """レビューテーブル"""
    __tablename__ = "reviews"

    review_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    rating = Column(Integer, nullable=False)
    review_title = Column(String(200))
    review_text = Column(Text)
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reviews")
    product = relationship("Product", back_populates="reviews")


def get_session(database_url: str = "sqlite:///test.db") -> Session:
    """セッションを取得する"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def create_user(session: Session, username: str, email: str, password_hash: str) -> User:
    """ユーザーを作成する"""
    user = User(username=username, email=email, password_hash=password_hash)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_products_by_category(session: Session, category_id: int) -> list[Product]:
    """カテゴリ別商品を取得する"""
    return session.query(Product).filter(
        Product.category_id == category_id,
        Product.is_available == True,
    ).order_by(Product.product_name).all()


if __name__ == "__main__":
    session = get_session()
    print("データベーステーブル作成完了")
