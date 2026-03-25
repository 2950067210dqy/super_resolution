"""
CSV 操作 start
"""
class CsvTable:
    """
    基于固定列名(CSV_COLUMNS)的 CSV CRUD 工具类。
    提供创建、读取、更新、删除与切换文件能力。
    """

    def __init__(self, file_path: str | Path, columns: list[str]):
        """
        初始化 CSV 表对象。

        功能：
        - 记录当前操作的 CSV 文件路径和列名
        - 自动创建父目录
        - 若文件不存在，按当前列名创建空表（仅表头）
        """
        self.file_path = Path(file_path)
        self.columns = columns
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_all([])

    def switch_file(self, file_path: str | Path) -> None:
        """
        切换到另一个 CSV 文件继续操作（列结构保持 self.columns）。

        功能：
        - 修改当前 file_path
        - 自动创建新文件父目录
        - 若目标文件不存在，创建带当前列名表头的空文件
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_all([])

    def _read_all(self) -> list[dict[str, str]]:
        """
        读取当前 CSV 文件全部数据并返回行列表。

        功能：
        - 使用 DictReader 按列名读取
        - 校验文件表头与 self.columns 完全一致
        - 返回 list[dict]（每行一个字典）
        """
        with self.file_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != self.columns:
                raise ValueError(f"CSV 列不匹配: {reader.fieldnames} != {self.columns}")
            return list(reader)

    def _write_all(self, rows: list[dict[str, Any]]) -> None:
        """
        覆盖写入当前 CSV 的全部数据。

        功能：
        - 先写表头（self.columns）
        - 再逐行写入 rows
        - 对缺失列自动补空字符串，保证列完整性
        """
        with self.file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for row in rows:
                safe_row = {col: row.get(col, "") for col in self.columns}
                writer.writerow(safe_row)

    def create(self, row: dict[str, Any]) -> None:
        """
        新增一行记录。

        功能：
        - 读取现有全部数据
        - 将新行按 self.columns 对齐后追加
        - 回写到 CSV 文件
        """
        rows = self._read_all()
        rows.append({col: row.get(col, "") for col in self.columns})
        self._write_all(rows)

    def read(
        self,
        where: Callable[[dict[str, str]], bool] | None = None
    ) -> list[dict[str, str]]:
        """
        查询记录。

        功能：
        - where=None：返回全部行
        - where 不为 None：返回满足条件的行
        """
        rows = self._read_all()
        if where is None:
            return rows
        return [r for r in rows if where(r)]

    def update(
        self,
        where: Callable[[dict[str, str]], bool],
        updates: dict[str, Any],
    ) -> int:
        """
        更新满足条件的记录。

        功能：
        - 遍历所有行，命中 where 的行执行字段更新
        - 仅更新在 self.columns 中存在的列
        - 回写文件并返回更新行数
        """
        rows = self._read_all()
        count = 0
        for r in rows:
            if where(r):
                for k, v in updates.items():
                    if k in self.columns:
                        r[k] = str(v)
                count += 1
        self._write_all(rows)
        return count

    def delete(self, where: Callable[[dict[str, str]], bool]) -> int:
        """
        删除满足条件的记录。

        功能：
        - 过滤掉命中 where 的行
        - 回写剩余数据
        - 返回删除行数
        """
        rows = self._read_all()
        new_rows = [r for r in rows if not where(r)]
        deleted = len(rows) - len(new_rows)
        self._write_all(new_rows)
        return deleted
"""
CSV 操作 end
"""