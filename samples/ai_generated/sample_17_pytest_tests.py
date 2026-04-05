"""
pytestテストスイート
Python基礎: テスト設計、フィクスチャ、パラメタライズ
DevOps語録: pytest, fixture, assert, mock, parametrize
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from typing import Optional


class Calculator:
    """テスト対象のクラス"""

    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("ゼロ除算エラー")
        return a / b

    def factorial(self, n: int) -> int:
        if n < 0:
            raise ValueError("負数のfactorialは未定義")
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)


class UserService:
    """ユーザーサービス（外部依存あり）"""

    def __init__(self, db_client):
        self.db = db_client

    def get_user(self, user_id: int) -> Optional[dict]:
        return self.db.find_one({"user_id": user_id})

    def create_user(self, username: str, email: str) -> dict:
        user = {"username": username, "email": email, "is_active": True}
        result = self.db.insert_one(user)
        user["user_id"] = result.inserted_id
        return user

    def deactivate_user(self, user_id: int) -> bool:
        result = self.db.update_one(
            {"user_id": user_id},
            {"$set": {"is_active": False}},
        )
        return result.modified_count > 0


# === フィクスチャ ===

@pytest.fixture
def calculator():
    """Calculatorインスタンスを提供する"""
    return Calculator()


@pytest.fixture
def mock_db():
    """モックDBクライアントを提供する"""
    return MagicMock()


@pytest.fixture
def user_service(mock_db):
    """UserServiceインスタンスを提供する"""
    return UserService(mock_db)


# === Calculator テスト ===

class TestCalculator:
    """Calculator クラスのテスト"""

    def test_add(self, calculator):
        """足し算のテスト"""
        result = calculator.add(2, 3)
        assert result == 5

    def test_add_negative(self, calculator):
        """負数の足し算"""
        result = calculator.add(-1, -2)
        assert result == -3

    @pytest.mark.parametrize("a, b, expected", [
        (10, 2, 5.0),
        (7, 3, 7 / 3),
        (0, 5, 0.0),
        (-6, 2, -3.0),
    ])
    def test_divide(self, calculator, a, b, expected):
        """除算のパラメタライズテスト"""
        result = calculator.divide(a, b)
        assert result == pytest.approx(expected)

    def test_divide_by_zero(self, calculator):
        """ゼロ除算で例外が発生する"""
        with pytest.raises(ValueError, match="ゼロ除算"):
            calculator.divide(1, 0)

    @pytest.mark.parametrize("n, expected", [
        (0, 1), (1, 1), (5, 120), (10, 3628800),
    ])
    def test_factorial(self, calculator, n, expected):
        """factorialの正常系テスト"""
        assert calculator.factorial(n) == expected

    def test_factorial_negative(self, calculator):
        """負数でValueError"""
        with pytest.raises(ValueError):
            calculator.factorial(-1)


# === UserService テスト（モック使用）===

class TestUserService:
    """UserService のテスト"""

    def test_get_user_found(self, user_service, mock_db):
        """ユーザー取得成功"""
        mock_db.find_one.return_value = {"user_id": 1, "username": "test"}
        result = user_service.get_user(1)
        assert result is not None
        assert result["username"] == "test"
        mock_db.find_one.assert_called_once_with({"user_id": 1})

    def test_get_user_not_found(self, user_service, mock_db):
        """ユーザーが見つからない"""
        mock_db.find_one.return_value = None
        result = user_service.get_user(999)
        assert result is None

    def test_create_user(self, user_service, mock_db):
        """ユーザー作成"""
        mock_db.insert_one.return_value = Mock(inserted_id=42)
        result = user_service.create_user("newuser", "new@example.com")
        assert result["username"] == "newuser"
        assert result["user_id"] == 42
        assert result["is_active"] is True

    def test_deactivate_user(self, user_service, mock_db):
        """ユーザー無効化"""
        mock_db.update_one.return_value = Mock(modified_count=1)
        assert user_service.deactivate_user(1) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
