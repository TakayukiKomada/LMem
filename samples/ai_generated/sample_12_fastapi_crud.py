"""
FastAPI CRUDアプリケーション
Python基礎: async/await、Pydantic、HTTPステータス
Web語録: FastAPI, HTTPException, BaseModel, CRUD
"""
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItemCreate(BaseModel):
    """アイテム作成リクエスト"""
    product_name: str = Field(..., min_length=1, max_length=200)
    product_price: float = Field(..., gt=0)
    product_description: Optional[str] = None
    category_id: int = Field(..., gt=0)
    stock_quantity: int = Field(default=0, ge=0)
    is_available: bool = True


class ItemResponse(BaseModel):
    """アイテムレスポンス"""
    product_id: int
    product_name: str
    product_price: float
    product_description: Optional[str]
    category_id: int
    stock_quantity: int
    is_available: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    """ページネーション付きレスポンス"""
    items: list[ItemResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


# ダミーDB
db_items: dict[int, dict] = {}
next_id = 1


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "EC API サーバー稼働中"}


@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate):
    """アイテムを作成する"""
    global next_id
    now = datetime.now()
    db_item = {
        "product_id": next_id,
        **item.model_dump(),
        "created_at": now,
        "updated_at": now,
    }
    db_items[next_id] = db_item
    next_id += 1
    return db_item


@app.get("/items", response_model=PaginatedResponse)
async def list_items(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    category_id: Optional[int] = None,
    search_query: Optional[str] = None,
    sort_by: str = Query(default="created_at"),
):
    """アイテム一覧を取得する"""
    items = list(db_items.values())

    if category_id is not None:
        items = [i for i in items if i["category_id"] == category_id]
    if search_query:
        items = [i for i in items if search_query.lower() in i["product_name"].lower()]

    total_count = len(items)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    items = items[offset:offset + page_size]

    return PaginatedResponse(
        items=items,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


@app.get("/items/{product_id}", response_model=ItemResponse)
async def get_item(product_id: int):
    """アイテムを取得する"""
    if product_id not in db_items:
        raise HTTPException(status_code=404, detail="商品が見つかりません")
    return db_items[product_id]


@app.delete("/items/{product_id}", status_code=204)
async def delete_item(product_id: int):
    """アイテムを削除する"""
    if product_id not in db_items:
        raise HTTPException(status_code=404, detail="商品が見つかりません")
    del db_items[product_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
