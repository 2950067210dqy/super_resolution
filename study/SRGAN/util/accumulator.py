
class Accumulator:
    """在n个变量上累加  通用累加器：用于在一个 epoch 内累计多个标量指标（如 G/D 各项 loss）。"""
    def __init__(self, n):
        """初始化 n 个累加槽位。"""
        self.data = [0.0] * n

    def add(self, *args):
        """逐项累加输入值到内部槽位。"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """将所有槽位清零。"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """按索引读取某个累计值。"""
        return self.data[idx]