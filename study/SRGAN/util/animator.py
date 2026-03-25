class Animator:
    """
    损失函数折线图  训练曲线可视化器：支持多序列记录、分组绘图、导出静态 PNG/动态 GIF。
    """
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=None,
        figsize=(6, 4),
    ):
        """初始化坐标轴配置、图例、样式以及帧缓存容器。"""

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend or []
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.base_figsize = figsize

        self.X = None
        self.Y = None
        self.frames = []

        self.fmts = fmts if fmts is not None else self._build_auto_fmts()

        self.fig = None
        self.axes = None

    def _build_auto_fmts(self):
        """自动生成足够数量的线型/颜色/标记组合。"""
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        linestyles = ["-", "--", "-.", ":"]
        markers = ["", "o", "s", "d", "^", "v", "x", "*"]

        n = len(self.legend) if self.legend else 8
        fmts = []

        for linestyle in linestyles:
            for marker in markers:
                for color in colors:
                    fmts.append(f"{color}{linestyle}{marker}")
                    if len(fmts) >= n:
                        return fmts

        return fmts

    def add(self, x, y):
        """追加一帧曲线数据到内部缓存（支持单值/多值输入）。"""

        if not hasattr(y, "__len__"):
            y = [y]
        y = list(y)

        n = len(self.legend) if self.legend else len(y)

        if not hasattr(x, "__len__"):
            x = [x] * n
        else:
            x = list(x)
            if len(x) < n:
                x = x + [x[-1] if x else None] * (n - len(x))
            else:
                x = x[:n]

        if len(y) < n:
            y = y + [None] * (n - len(y))
        else:
            y = y[:n]

        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.frames.append(([row[:] for row in self.X], [row[:] for row in self.Y]))

    def _filter_series(self, Xf, Yf, exclude_legends=None):
        """按图例过滤不需要显示的序列，并打包绘图元信息。"""

        exclude_legends = set(exclude_legends or [])
        filtered = []

        n = max(len(self.legend), len(Xf), len(Yf))
        for i in range(n):
            xx = Xf[i] if i < len(Xf) else []
            yy = Yf[i] if i < len(Yf) else []
            name = self.legend[i] if i < len(self.legend) else f"series_{i}"
            if name in exclude_legends:
                continue
            filtered.append({
                "idx": i,
                "name": name,
                "x": xx,
                "y": yy,
                "fmt": self.fmts[i % len(self.fmts)],
            })
        return filtered

    def _series_scale(self, y):
        """计算单条序列的尺度（绝对值最大值），用于分组。"""
        if not y:
            return 0.0
        return max(abs(v) for v in y)

    def _group_series_by_scale(self, series_list, split_ratio=8.0):
        """按量级自动分组，避免大/小量级曲线互相遮蔽。"""
        non_empty = []
        empty = []

        for s in series_list:
            scale = self._series_scale(s["y"])
            item = {**s, "scale": scale}
            if scale == 0:
                empty.append(item)
            else:
                non_empty.append(item)

        non_empty.sort(key=lambda s: s["scale"])

        groups = []
        current_group = []

        for s in non_empty:
            if not current_group:
                current_group.append(s)
                continue

            current_max = max(item["scale"] for item in current_group)
            if current_max == 0 or s["scale"] / current_max <= split_ratio:
                current_group.append(s)
            else:
                groups.append(current_group)
                current_group = [s]

        if current_group:
            groups.append(current_group)

        if empty:
            if groups:
                groups[0].extend(empty)
            else:
                groups = [empty]

        return groups if groups else [[]]

    def _apply_fixed_groups(self, series_list, fixed_groups=None, split_ratio=8.0):
        """
          应用用户指定分组（支持重复引用同一label），
        其余序列再走自动分组逻辑。
        fixed_groups 支持:
        - 按名字: "g_loss"
        - 按名字+序号: "g_loss#0" / "g_loss#1"（同名序列时可精确指定）
        - 按索引: 0, 1, 2 ...
        """
        fixed_groups = fixed_groups or []

        def clone_series(src):
            return {
                "idx": src["idx"],
                "name": src["name"],
                "x": list(src["x"]),
                "y": list(src["y"]),
                "fmt": src["fmt"],
            }

        # name -> [series...]
        by_name = {}
        for s in series_list:
            by_name.setdefault(s["name"], []).append(s)

        final_groups = []
        used_positions = set()

        for group in fixed_groups:
            cur = []
            for item in group:
                src = None

                # 1) 整数索引
                if isinstance(item, int):
                    if 0 <= item < len(series_list):
                        src = series_list[item]
                        used_positions.add(item)

                # 2) 字符串：name 或 name#k
                elif isinstance(item, str):
                    if "#" in item:
                        name, k = item.rsplit("#", 1)
                        if k.isdigit():
                            k = int(k)
                            candidates = by_name.get(name, [])
                            if candidates:
                                src = candidates[min(k, len(candidates) - 1)]
                    else:
                        candidates = by_name.get(item, [])
                        if candidates:
                            src = candidates[0]

                    if src is not None:
                        for pos, s in enumerate(series_list):
                            if s is src:
                                used_positions.add(pos)
                                break

                if src is not None:
                    cur.append(clone_series(src))

            if cur:
                final_groups.append(cur)

        # 剩余未固定的再动态分组
        remaining = [s for i, s in enumerate(series_list) if i not in used_positions]
        dynamic_groups = self._group_series_by_scale(remaining, split_ratio=split_ratio)
        final_groups.extend([g for g in dynamic_groups if g])

        return final_groups if final_groups else [[]]

    def _build_figure(self, n_subplots):
        """按子图数量创建/重建画布。"""

        if self.fig is not None:
            plt.close(self.fig)

        width, height = self.base_figsize
        self.fig, axes = plt.subplots(
            n_subplots,
            1,
            figsize=(width, height * n_subplots),
            squeeze=False
        )
        self.axes = [ax[0] for ax in axes]

    def _config_axis(self, ax, series_group):
        """为单子图设置坐标轴范围、比例尺、图例和网格。"""
        ax.set_xlabel(self.xlabel or "")
        ax.set_ylabel(self.ylabel or "")

        if self.xlim is not None:
            ax.set_xlim(self.xlim)

        y_values = [v for s in series_group for v in s["y"]]
        if y_values:
            data_min = min(y_values)
            data_max = max(y_values)

            if self.ylim is not None:
                ymin, ymax = self.ylim
                ymax = max(ymax, data_max)
                ymin = min(ymin, data_min)
            else:
                ymin, ymax = data_min, data_max

            if ymin == ymax:
                pad = 1.0 if ymin == 0 else abs(ymin) * 0.05
                ymin -= pad
                ymax += pad

            ax.set_ylim((ymin, ymax+0.2))

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        if series_group:
            handles = ax.lines
            labels = [s["name"] for s in series_group]
            ax.legend(handles, labels, loc="upper right", fontsize=9)

        ax.grid(True, alpha=0.3)

    def _draw_frame(self, frame_idx, exclude_legends=None, split_ratio=8.0, fixed_groups=None):
        """渲染指定帧到画布。"""
        Xf, Yf = self.frames[frame_idx]
        filtered_series = self._filter_series(Xf, Yf, exclude_legends=exclude_legends)
        groups = self._apply_fixed_groups(
            filtered_series,
            fixed_groups=fixed_groups,
            split_ratio=split_ratio
        )

        self._build_figure(len(groups))

        for ax, group in zip(self.axes, groups):
            ax.cla()
            for s in group:
                ax.plot(s["x"], s["y"], s["fmt"])
            self._config_axis(ax, group)

        self.fig.tight_layout()

    def save(self, gif_path="train.gif", fps=20, exclude_legends=None, split_ratio=8.0, fixed_groups=None):
        """导出训练过程动态 GIF。"""
        if not self.frames:
            raise ValueError("没有可保存的帧，请先调用 add().")

        self._draw_frame(
            0,
            exclude_legends=exclude_legends,
            split_ratio=split_ratio,
            fixed_groups=fixed_groups,
        )

        def update(frame_idx):
            self._draw_frame(
                frame_idx,
                exclude_legends=exclude_legends,
                split_ratio=split_ratio,
                fixed_groups=fixed_groups,
            )
            lines = []
            for ax in self.axes:
                lines.extend(ax.lines)
            return lines

        ani = FuncAnimation(
            self.fig,
            update,
            frames=len(self.frames),
            interval=1000 / fps,
            blit=False
        )
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(self.fig)
        self.fig = None
        self.axes = None

    def save_png(self, png_path="train.png", exclude_legends=None, split_ratio=8.0, fixed_groups=None):
        """导出最后一帧静态 PNG。"""
        if not self.frames:
            raise ValueError("没有可保存的帧，请先调用 add().")

        self._draw_frame(
            len(self.frames) - 1,
            exclude_legends=exclude_legends,
            split_ratio=split_ratio,
            fixed_groups=fixed_groups,
        )
        self.fig.savefig(png_path, dpi=200, bbox_inches="tight")