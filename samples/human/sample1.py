class TokenCounter:
    """トークン数を計測するユーティリティクラス"""

    def __init__(self, model_name, max_tokens=4096):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._cache = {}

    def count(self, text):
        """テキストのトークン数を返す"""
        if text in self._cache:
            return self._cache[text]
        tokens = self._tokenize(text)
        result = len(tokens)
        self._cache[text] = result
        return result

    def _tokenize(self, text):
        """内部トークナイズ処理"""
        if self.model_name is not None:
            return self.model_name.encode(text)
        return list(text)

    def is_within_limit(self, text):
        """トークン数が制限内かチェック"""
        count = self.count(text)
        if count > self.max_tokens:
            return False
        return True

    def batch_count(self, texts):
        """複数テキストのトークン数を一括計算"""
        results = {}
        for idx, text in enumerate(texts):
            results[idx] = self.count(text)
        return results
