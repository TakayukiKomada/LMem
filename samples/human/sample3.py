class DataPipeline:
    """データ処理パイプライン"""

    def __init__(self, steps=None):
        self.steps = steps or []
        self._results = []

    def add_step(self, func, name=None):
        """処理ステップを追加"""
        self.steps.append({
            "func": func,
            "name": name or func.__name__,
        })
        return self

    def run(self, data):
        """パイプラインを実行"""
        self._results = []
        current = data
        for step in self.steps:
            try:
                current = step["func"](current)
                self._results.append({
                    "step": step["name"],
                    "status": "success",
                    "output_size": len(current) if hasattr(current, "__len__") else None,
                })
            except Exception as e:
                self._results.append({
                    "step": step["name"],
                    "status": "error",
                    "error": str(e),
                })
                raise
        return current

    def get_report(self):
        """実行レポートを返す"""
        if not self._results:
            return "No results available"
        lines = []
        for r in self._results:
            if r["status"] == "success":
                lines.append(f"  [OK] {r['step']}: size={r['output_size']}")
            else:
                lines.append(f"  [NG] {r['step']}: {r['error']}")
        return "\n".join(lines)
