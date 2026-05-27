from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zipfile
from dataclasses import dataclass
from html import escape as xml_escape
from pathlib import Path
from typing import Iterable

# numpy / matplotlib 都在真正绘图时延迟导入。
# 这样 --list-only 只检查目录映射，不会因为当前 Python 环境暂时缺绘图库而失败。
np = None
plt = None


def ensure_numpy():
    """延迟导入 numpy；真实读写 npy 或绘图前必须可用。"""

    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def ensure_matplotlib():
    """延迟导入 matplotlib，并固定 Agg 后端，适合无界面环境批量保存图片。"""

    global plt
    ensure_numpy()
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt

try:
    from .global_class import global_data
except ImportError:
    # 允许用户在 SRGAN 根目录直接执行：python all_handle/pipeline.py
    from global_class import global_data


@dataclass(frozen=True)
class GroupContext:
    """一个绘图分组，对应 class_1/class_2 + predict_all/test_all + 某个类别。"""

    comparison_name: str
    experiment_keys: tuple[str, ...]
    class_name: str
    split_name: str
    category_name: str
    experiment_dirs: dict[str, Path]

    @property
    def tag(self) -> str:
        return "_".join(
            safe_name(part)
            for part in (self.comparison_name, self.class_name, self.split_name, self.category_name)
            if part
        )


@dataclass(frozen=True)
class SampleBundle:
    """同一个样本在不同实验目录中的位置集合。"""

    sample_name: str
    sample_dirs: dict[str, Path]


def normalize_name(name: str) -> str:
    """把目录名归一化，解决 class_1/class1、Backstep/backstep 这类大小写和下划线差异。"""

    return str(name).replace("-", "_").replace(" ", "_").lower()


def safe_name(name: str) -> str:
    """把任意类别/样本名转换成适合作为输出文件名的安全字符串。"""

    text = str(name).strip().replace("\\", "_").replace("/", "_").replace(":", "_")
    return text or "none"


def finite_values(array: np.ndarray) -> np.ndarray:
    """提取有限数值，统一跳过 NaN/Inf，避免个别坏值把色条和坐标轴拉坏。"""

    arr = np.asarray(array, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def ensure_2d_image(array: np.ndarray) -> np.ndarray:
    """把常见的 BCHW/CHW/HWC/单通道图像整理成 Matplotlib 可显示的二维或 HWC 数组。"""

    arr = np.asarray(array)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def load_npy(path: Path) -> np.ndarray | None:
    """安全读取 npy；失败时返回 None，由调用方决定跳过还是回退到 png。"""

    if not path.exists():
        return None
    numpy = ensure_numpy()
    try:
        return numpy.load(path, allow_pickle=True)
    except Exception:
        return None


def load_png(path: Path) -> np.ndarray | None:
    """读取已有 png 作为拼图回退方案；原始 npy 不存在时仍可生成论文排版图。"""

    if not path.exists():
        return None
    pyplot = ensure_matplotlib()
    try:
        return pyplot.imread(str(path))
    except Exception:
        return None


def first_existing(directory: Path, names: Iterable[str]) -> Path | None:
    """在目录中按候选文件名顺序寻找第一个存在的文件。"""

    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def array_to_xy(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """将能量谱或直方图 npy 统一转换为 x/y 两列数据。"""

    arr = np.asarray(array)
    if arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            x = obj.get("x") or obj.get("centers") or obj.get("bin_centers")
            y = obj.get("y") or obj.get("counts") or obj.get("values")
            if x is not None and y is not None:
                return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 2:
        return arr[0].astype(np.float64), arr[1].astype(np.float64)
    if arr.ndim == 1:
        y = arr.astype(np.float64)
        x = np.arange(1, y.size + 1, dtype=np.float64)
        return x, y
    return None


def flow_to_hw2(array: np.ndarray) -> np.ndarray | None:
    """把不同保存格式的光流数组整理成 H x W x 2，后续统一取 u/v/s。"""

    arr = np.asarray(array)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[..., :2].astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return np.moveaxis(arr[:2], 0, -1).astype(np.float32)
    return None


class AllHandlePipeline:
    """读取八个实验已有结果，生成跨实验对比图。"""

    def __init__(self, cfg: type[global_data.all_handle], enable_plotting: bool = True):
        self.cfg = cfg
        self.warnings: list[str] = []
        # 当前正在生成的对比组；用于决定图例 label、legend 顺序和输出子目录。
        self.active_comparison_name = "eight_experiments"
        self.output_root_dir = self.resolve_output_root_dir()
        self.output_root_dir.mkdir(parents=True, exist_ok=True)
        # 运行时间记录用于命令行进度显示和 summary.json，便于全量生成时定位耗时最长的类别/步骤。
        self.run_started_at: float | None = None
        self.timing_records: list[dict[str, str | float | int]] = []
        if enable_plotting:
            self._setup_matplotlib()

    def resolve_output_root_dir(self) -> Path:
        """解析输出根目录；全局 OUTPUT_ROOT_DIR 为 None 时保持默认 all_handle/output。"""

        configured = getattr(self.cfg, "OUTPUT_ROOT_DIR", None)
        if configured is None:
            return Path(getattr(self.cfg, "DEFAULT_OUTPUT_ROOT_DIR", Path(__file__).resolve().parent / "output"))
        return Path(configured)

    def _setup_matplotlib(self) -> None:
        """统一字体、字号和 SVG 输出方式，保证不同图之间风格一致。"""

        pyplot = ensure_matplotlib()
        pyplot.rcParams["font.family"] = self.select_available_font_family()
        pyplot.rcParams["font.size"] = self.cfg.TICK_LABEL_SIZE
        pyplot.rcParams["axes.labelsize"] = self.cfg.AXIS_LABEL_SIZE
        pyplot.rcParams["legend.fontsize"] = self.cfg.LEGEND_FONT_SIZE
        pyplot.rcParams["svg.fonttype"] = "none"
        pyplot.rcParams["axes.unicode_minus"] = False

    def select_available_font_family(self) -> str:
        """按全局字体优先级选择本机已安装字体，避免缺 Times New Roman 时产生大量 warning。"""

        pyplot = ensure_matplotlib()
        font_manager = pyplot.matplotlib.font_manager
        configured = getattr(self.cfg, "FONT_FAMILY", "")
        candidates: list[str] = []
        if isinstance(configured, (tuple, list)):
            candidates.extend(str(item) for item in configured if item)
        elif configured:
            candidates.append(str(configured))
        candidates.extend(str(item) for item in getattr(self.cfg, "FONT_FAMILY_FALLBACKS", ()) if item)

        installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
        generic_families = {"serif", "sans-serif", "monospace", "cursive", "fantasy"}
        for family in candidates:
            # 先使用真实安装字体；若候选是 Matplotlib 通用字体族，也允许直接使用。
            if family in installed_fonts or family in generic_families:
                return family
        return "DejaVu Serif"

    def warn(self, message: str) -> None:
        """记录缺文件或无法解析的情况，脚本结束时写入 summary。"""

        self.warnings.append(message)
        print(f"[all_handle] {message}")

    def progress(self, message: str) -> None:
        """轻量进度输出；不依赖 tqdm，适合 Windows/服务器普通命令行。"""

        if not getattr(self.cfg, "PROGRESS_ENABLED", True):
            return
        prefix = getattr(self.cfg, "PROGRESS_PREFIX", "[all_handle]")
        print(f"{prefix} {message}", flush=True)

    def format_duration(self, seconds: float) -> str:
        """把秒数格式化成易读运行时间，所有进度日志和 summary 复用同一规则。"""

        decimals = int(getattr(self.cfg, "PROGRESS_RUNTIME_DECIMALS", 2))
        seconds = max(0.0, float(seconds))
        if seconds < 60.0:
            return f"{seconds:.{decimals}f}s"
        minutes, remain = divmod(seconds, 60.0)
        if minutes < 60.0:
            return f"{int(minutes)}m {remain:.{decimals}f}s"
        hours, minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h {int(minutes)}m {remain:.{decimals}f}s"

    def progress_step(
        self,
        group_index: int,
        group_total: int,
        step_index: int,
        step_total: int,
        step_name: str,
    ) -> None:
        """输出当前 group 内部步骤进度，便于定位长时间运行卡在哪个绘图阶段。"""

        if not getattr(self.cfg, "PROGRESS_SHOW_STEPS", True):
            return
        self.progress(f"group {group_index}/{group_total} step {step_index}/{step_total}: {step_name}")

    def progress_step_done(
        self,
        group_index: int,
        group_total: int,
        step_index: int,
        step_total: int,
        step_name: str,
        elapsed_seconds: float,
    ) -> None:
        """输出单个 step 的耗时，方便判断是直方图、误差图还是组合图更慢。"""

        if not getattr(self.cfg, "PROGRESS_SHOW_STEPS", True):
            return
        if not getattr(self.cfg, "PROGRESS_SHOW_RUNTIME", True):
            return
        self.progress(
            f"group {group_index}/{group_total} step {step_index}/{step_total} done: "
            f"{step_name} in {self.format_duration(elapsed_seconds)}"
        )

    def resume_step_start_index(self, group_index: int, steps: list[tuple[str, object]]) -> int:
        """根据全局续跑参数决定当前 group 从哪个 step 开始；返回 0 表示从第一步跑。"""

        resume_group = getattr(self.cfg, "RESUME_GROUP_INDEX", None)
        if resume_group is None:
            return 0
        try:
            resume_group = int(resume_group)
        except (TypeError, ValueError):
            return 0
        if group_index < resume_group:
            return len(steps)
        if group_index > resume_group:
            return 0

        # 到达指定 group 后，可以用 step 名称或 step 序号继续；名称优先，避免步骤增删后序号漂移。
        resume_step_name = getattr(self.cfg, "RESUME_STEP_NAME", None)
        if resume_step_name:
            target = normalize_name(str(resume_step_name))
            for idx, (step_name, _step_func) in enumerate(steps):
                if normalize_name(step_name) == target:
                    return idx
            self.warn(f"resume step name not found in group {group_index}: {resume_step_name}; start from first step.")
            return 0

        resume_step = getattr(self.cfg, "RESUME_STEP_INDEX", None)
        if resume_step is None:
            return 0
        try:
            resume_step = int(resume_step)
        except (TypeError, ValueError):
            return 0
        return max(0, min(len(steps), resume_step - 1))

    def enabled_output_stages(self) -> set[str]:
        """读取全局输出阶段过滤参数，返回本次需要生成的 01/02/03/04 阶段集合。"""

        configured = getattr(self.cfg, "OUTPUT_STAGE_FILTER", None)
        all_stages = {
            "energy_spectrum",
            "error_maps",
            "error_histograms",
            "composite_panels",
            "tbl_profile_overlay",
            "particle_stats_metrics",
            "flow_u_epe_hist_overlay",
            "tbl_02_error_map",
        }
        if configured is None:
            return all_stages
        if isinstance(configured, str):
            raw_items = [configured]
        else:
            try:
                raw_items = list(configured)
            except TypeError:
                raw_items = [configured]
        aliases = getattr(self.cfg, "OUTPUT_STAGE_ALIASES", {})
        enabled: set[str] = set()
        for item in raw_items:
            key = normalize_name(str(item))
            stage = aliases.get(key, key)
            if stage == "all":
                return all_stages
            if stage in all_stages:
                enabled.add(stage)
            else:
                self.warn(f"unknown output stage ignored: {item}")
        return enabled or all_stages

    def group_progress_label(self, group: GroupContext) -> str:
        """把当前分组压缩成一行可读文本，避免进度输出里出现很长的路径。"""

        return f"{group.comparison_name}/{group.class_name}/{group.split_name}/{group.category_name}"

    def experiment_label(self, exp_key: str) -> str:
        # 同一个实验在不同对比组中可能需要不同图例文字：
        # 八组对比中 PIV_A_Esrgan_v4 显示为 ESRuRAFT-PIV，倍率对比中显示为 ESRuRAFT-PIV x4。
        group_labels = getattr(self.cfg, "COMPARISON_GROUP_LABELS", {}).get(self.active_comparison_name, {})
        return group_labels.get(exp_key, self.cfg.EXPERIMENT_LABELS.get(exp_key, exp_key))

    def experiment_color(self, exp_key: str) -> str:
        return self.cfg.EXPERIMENT_COLORS.get(exp_key, "#333333")

    def experiment_hist_color(self, exp_key: str) -> str:
        # 误差直方图使用独立调色板：半透明柱子叠加后更容易混色，
        # 单独取色可以保证八组/倍率对比的颜色既明显区分，又保持论文常用配色风格。
        return getattr(self.cfg, "EXPERIMENT_HIST_COLORS", {}).get(exp_key, self.experiment_color(exp_key))

    def darken_color(self, color: str, factor: float | None = None) -> str:
        """把直方图填充色加深，用作柱子边框和图例边框，避免图例只有浅色块而辨识度不足。"""

        ensure_matplotlib()
        factor = float(getattr(self.cfg, "HIST_EDGE_DARKEN", 0.72) if factor is None else factor)
        factor = max(0.0, min(1.0, factor))
        rgb = plt.matplotlib.colors.to_rgb(color)
        dark = tuple(max(0.0, min(1.0, channel * factor)) for channel in rgb)
        return plt.matplotlib.colors.to_hex(dark)

    def apply_hist_legend(self, ax: plt.Axes, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """误差直方图专用 legend：面色半透明，边框使用同一实验的深色粗线。"""

        ensure_matplotlib()
        patch_cls = plt.matplotlib.patches.Patch
        ordered_keys = [exp_key for exp_key in self.legend_order_keys() if exp_key in series]
        ordered_keys.extend(exp_key for exp_key in series if exp_key not in ordered_keys)
        handles = []
        labels = []
        for exp_key in ordered_keys:
            color = self.experiment_hist_color(exp_key)
            edge_width = float(getattr(self.cfg, "HIST_LEGEND_EDGE_LINE_WIDTH", 0.0))
            handles.append(
                patch_cls(
                    facecolor=color,
                    edgecolor="none" if edge_width <= 0 else self.darken_color(color),
                    linewidth=edge_width,
                    alpha=self.cfg.HIST_ALPHA,
                )
            )
            labels.append(self.experiment_label(exp_key))
        if handles:
            ax.legend(handles, labels, frameon=False)

    def particle_stats_bar_color(self, exp_key: str | None) -> str:
        """颗粒统计条形图颜色：GT 与每个实验固定颜色，图例和柱子保持一致。"""

        if exp_key is None:
            return getattr(self.cfg, "PARTICLE_STATS_GT_BAR_COLOR", "#666666")
        color_map = getattr(self.cfg, "PARTICLE_STATS_EXPERIMENT_BAR_COLORS", {})
        return color_map.get(exp_key, getattr(self.cfg, "PARTICLE_STATS_BAR_COLOR", self.experiment_color(exp_key)))

    def format_particle_stats_value(self, value: float) -> str:
        """颗粒统计柱顶数值强制使用普通小数/整数，避免 Matplotlib 或 .3g 产生科学计数法。"""

        decimals = int(getattr(self.cfg, "PARTICLE_STATS_VALUE_DECIMALS", 4))
        if not math.isfinite(value):
            return ""
        if abs(value - round(value)) < 10 ** (-(decimals + 1)):
            return str(int(round(value)))
        text = f"{value:.{decimals}f}"
        return text.rstrip("0").rstrip(".")

    def legend_order_keys(self) -> tuple[str, ...]:
        """读取固定 legend 顺序；没有单独配置时回退到实验顺序。"""

        group_order = getattr(self.cfg, "COMPARISON_GROUP_LEGEND_ORDER", {}).get(self.active_comparison_name)
        if group_order:
            return tuple(group_order)
        return getattr(self.cfg, "LEGEND_EXPERIMENT_ORDER", self.cfg.EXPERIMENT_KEYS)

    def apply_ordered_legend(self, ax: plt.Axes, include_gt: bool = False, energy_style: bool = False) -> None:
        """
        按全局配置重排 legend。
        直方图为了遮挡关系会调整绘制顺序，但 legend 必须按论文指定顺序从上到下展示。
        """

        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        label_to_handle = {}
        for handle, label in zip(handles, labels):
            label_to_handle.setdefault(label, handle)

        ordered_labels = [
            self.experiment_label(exp_key)
            for exp_key in self.legend_order_keys()
            if self.experiment_label(exp_key) in label_to_handle
        ]
        if include_gt and self.cfg.GT_ENERGY_LABEL in label_to_handle:
            ordered_labels.append(self.cfg.GT_ENERGY_LABEL)
        for label in labels:
            if label not in ordered_labels:
                ordered_labels.append(label)
        legend = ax.legend(
            [label_to_handle[label] for label in ordered_labels],
            ordered_labels,
            frameon=bool(getattr(self.cfg, "ENERGY_LEGEND_FRAME", True)) if energy_style else False,
            fontsize=float(getattr(self.cfg, "ENERGY_LEGEND_FONT_SIZE", self.cfg.LEGEND_FONT_SIZE))
            if energy_style
            else self.cfg.LEGEND_FONT_SIZE,
        )
        if energy_style and legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(getattr(self.cfg, "ENERGY_LEGEND_FACE_COLOR", "#E6E6E6"))
            frame.set_edgecolor(getattr(self.cfg, "ENERGY_LEGEND_EDGE_COLOR", "#808080"))
            frame.set_alpha(float(getattr(self.cfg, "ENERGY_LEGEND_ALPHA", 0.58)))

    def experiment_root(self, exp_key: str, class_name: str, split_name: str) -> Path:
        """根据全局映射拼出某个实验、某个 class、某个 split 的结果根目录。"""

        exp_dir = self.cfg.EXPERIMENT_DIR_NAMES[exp_key]
        # x4/x8 的输出目录倍率不同，因此倍率目录按实验 key 单独读取；
        # 没配置的旧实验仍使用 SCALE_DIR_NAME，保证兼容之前八组对比实验。
        scale_dir = self.cfg.EXPERIMENT_SCALE_DIR_NAMES.get(exp_key, self.cfg.SCALE_DIR_NAME)
        class_norm = normalize_name(class_name)
        mode_dir = (
            "problem_class2_raft_piv"
            if class_norm in ("class_2", "class2")
            else "mixed_all_classes"
        )
        # 新增实验目录可能使用长目录名，也可能直接使用 bicubic_searaft/swinir_raft 这样的短目录名；
        # 这里按主目录名和别名依次尝试，找到已存在路径就返回，全部不存在时返回主目录路径用于后续跳过。
        candidate_dir_names = [exp_dir]
        for alias in getattr(self.cfg, "EXPERIMENT_DIR_NAME_ALIASES", {}).get(exp_key, ()):
            if alias not in candidate_dir_names:
                candidate_dir_names.append(alias)
        candidates = [
            self.cfg.DATA_ROOT_DIR
            / dir_name
            / class_name
            / mode_dir
            / self.cfg.RAFT_DIR_NAME
            / scale_dir
            / split_name
            for dir_name in candidate_dir_names
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def output_dir(self, *parts: str) -> Path:
        # 不同对比组必须分开输出，避免八组对比图、去除 widim/hs 的补充图与 x4/x8 倍率对比图混在一起。
        path = self.output_root_dir.joinpath(
            safe_name(self.active_comparison_name),
            *(safe_name(p) for p in parts),
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_figure(self, fig: plt.Figure, path_without_suffix: Path) -> None:
        """按全局格式保存 png/svg；图中不设置标题，只保存坐标轴和面板 label。"""

        path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in self.cfg.FIG_FORMATS:
            fig.savefig(
                path_without_suffix.with_suffix(f".{suffix}"),
                dpi=self.cfg.FIG_DPI,
                bbox_inches="tight",
                pad_inches=0.03,
            )
        plt.close(fig)

    def save_npy(self, path: Path, data) -> None:
        # 按全局开关控制是否在输出目录额外保存 npy 汇总文件；
        # 关闭后不影响读取原始实验结果中的 npy，只跳过 pipeline 自己生成的 .npy 输出。
        if not getattr(self.cfg, "SAVE_NPY_OUTPUTS", False):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data, allow_pickle=True)

    # =========================
    # 目录发现
    # =========================
    def discover_groups(self) -> list[GroupContext]:
        """扫描八个实验已有的 class/split/category 目录，生成后续绘图任务。"""

        groups: list[GroupContext] = []
        category_filter = None
        if self.cfg.CATEGORY_FILTER:
            category_filter = {normalize_name(v) for v in self.cfg.CATEGORY_FILTER}

        comparison_groups = getattr(
            self.cfg,
            "COMPARISON_GROUPS",
            {"eight_experiments": tuple(self.cfg.EXPERIMENT_KEYS)},
        )
        min_counts = getattr(self.cfg, "COMPARISON_GROUP_MIN_EXPERIMENTS", {})

        for comparison_name, experiment_keys in comparison_groups.items():
            experiment_keys = tuple(experiment_keys)
            min_count = int(min_counts.get(comparison_name, 1))
            for class_name in self.cfg.CLASS_NAMES:
                for split_name in self.cfg.SPLIT_NAMES:
                    split_roots = {
                        exp_key: self.experiment_root(exp_key, class_name, split_name)
                        for exp_key in experiment_keys
                    }
                    existing_roots = {
                        exp_key: root for exp_key, root in split_roots.items() if root.exists()
                    }
                    if len(existing_roots) < min_count:
                        continue

                    groups.append(
                        GroupContext(
                            comparison_name=comparison_name,
                            experiment_keys=experiment_keys,
                            class_name=class_name,
                            split_name=split_name,
                            category_name="all",
                            experiment_dirs=existing_roots,
                        )
                    )

                    category_map: dict[str, str] = {}
                    for root in existing_roots.values():
                        for child in root.iterdir():
                            if child.is_dir():
                                norm = normalize_name(child.name)
                                if norm in ("flow", "images"):
                                    continue
                                category_map.setdefault(norm, child.name)

                    for norm_name, display_name in sorted(category_map.items()):
                        if category_filter and norm_name not in category_filter:
                            continue
                        dirs = {}
                        for exp_key, root in existing_roots.items():
                            match = self.find_child_dir(root, display_name)
                            if match is not None:
                                dirs[exp_key] = match
                        if len(dirs) >= min_count:
                            groups.append(
                                GroupContext(
                                    comparison_name=comparison_name,
                                    experiment_keys=experiment_keys,
                                    class_name=class_name,
                                    split_name=split_name,
                                    category_name=display_name,
                                    experiment_dirs=dirs,
                                )
                            )
        return groups

    def find_child_dir(self, root: Path, name: str) -> Path | None:
        """按归一化名字寻找类别目录，避免 DNS_turbulence/dns_turb 这类大小写问题。"""

        target = normalize_name(name)
        for child in root.iterdir():
            if child.is_dir() and normalize_name(child.name) == target:
                return child
        return None

    def discover_sample_dirs(self, category_dir: Path, kind: str) -> dict[str, Path]:
        """
        发现某一类图需要的样本目录。
        predict_all 通常是 category/sample/previous，test_all 大图通常是 category/images/sample 或 category/flow/sample。
        """

        sample_dirs: dict[str, Path] = {}
        roots: list[Path] = []
        if kind in ("particle", "particle_stats"):
            roots.extend([category_dir, category_dir / "images"])
        elif kind in ("flow", "vorticity"):
            roots.extend([category_dir, category_dir / "flow"])
        else:
            roots.append(category_dir)

        for root in roots:
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and self.sample_dir_has_kind(child, kind):
                    sample_dirs.setdefault(child.name, child)
        return sample_dirs

    def sample_dir_has_kind(self, sample_dir: Path, kind: str) -> bool:
        """判断一个目录是否包含指定图类所需的关键文件。"""

        if kind == "particle":
            return any(
                (
                    self.particle_path(sample_dir, t, "error").exists()
                    or self.particle_image_path(sample_dir, t, "sr").exists()
                    or self.particle_crop_path(sample_dir, t, "error").exists()
                )
                for t in ("previous", "next")
            )
        if kind == "particle_stats":
            return any(
                self.particle_path(sample_dir, t, "stats").exists()
                or self.particle_path(sample_dir, t, "hist").exists()
                or self.particle_crop_path(sample_dir, t, "stats").exists()
                or self.particle_crop_path(sample_dir, t, "hist").exists()
                for t in ("previous", "next")
            )
        if kind == "flow":
            candidates = (
                "fake_flo.npy",
                "hr_flo.npy",
                "delta_u.npy",
                "delta_v.npy",
                "delta_s.npy",
                "delta_uvw.npy",
                getattr(self.cfg, "FLOW_UVS_COMPARE_FILE_NAME", "uvs_compare.png"),
            )
            return any((sample_dir / name).exists() for name in candidates)
        if kind == "vorticity":
            candidates = ("pred_vorticity.npy", "gt_vorticity.npy", "delta_vorticity.npy")
            return any((sample_dir / name).exists() for name in candidates)
        return False

    def bundle_samples(self, group: GroupContext, kind: str) -> list[SampleBundle]:
        """把同名样本在八个实验中的目录合并，便于统一色条和拼版。"""

        per_experiment = {
            exp_key: self.discover_sample_dirs(category_dir, kind)
            for exp_key, category_dir in group.experiment_dirs.items()
        }
        names: set[str] = set()
        for sample_map in per_experiment.values():
            names.update(sample_map.keys())

        if self.cfg.SAMPLE_FILTER:
            allowed = {normalize_name(v) for v in self.cfg.SAMPLE_FILTER}
            names = {name for name in names if normalize_name(name) in allowed}

        bundles: list[SampleBundle] = []
        for name in sorted(names):
            dirs = {
                exp_key: sample_map[name]
                for exp_key, sample_map in per_experiment.items()
                if name in sample_map
            }
            if dirs:
                bundles.append(SampleBundle(sample_name=name, sample_dirs=dirs))
        return bundles

    # =========================
    # 路径兼容层
    # =========================
    def particle_path(self, sample_dir: Path, time_name: str, key: str) -> Path:
        """兼容 previous/sr_error.npy 与 prev_sr_error.npy 两种颗粒结果保存方式。"""

        nested_name = {
            "lr": "lr.npy",
            "gt": "hr.npy",
            "sr": "fake.npy",
            "error": "sr_error.npy",
            "hist": "particle_binary_stats_hist.npy",
            "gt_binary": "particle_binary_stats_gt_binary.npy",
            "sr_binary": "particle_binary_stats_pred_binary.npy",
            "stats": "particle_binary_stats_stats.npy",
            "threshold": "particle_binary_stats_threshold.txt",
        }[key]
        prefix = "prev" if time_name == "previous" else "next"
        flat_name = {
            "lr": f"{prefix}_lr.npy",
            "gt": f"{prefix}_hr.npy",
            "sr": f"{prefix}_sr.npy",
            "error": f"{prefix}_sr_error.npy",
            "hist": f"{prefix}_particle_binary_stats_hist.npy",
            "gt_binary": f"{prefix}_particle_binary_stats_gt_binary.npy",
            "sr_binary": f"{prefix}_particle_binary_stats_pred_binary.npy",
            "stats": f"{prefix}_particle_binary_stats_stats.npy",
            "threshold": f"{prefix}_particle_binary_stats_threshold.txt",
        }[key]
        nested = sample_dir / time_name / nested_name
        return nested if nested.exists() else sample_dir / flat_name

    def particle_crop_path(self, sample_dir: Path, time_name: str, key: str) -> Path:
        """读取 TBL 已保存的 crop 颗粒结果；SR/GT/LR crop 没有 npy 时由 pipeline.py 从 full-frame 裁。"""

        prefix = "prev" if time_name == "previous" else "next"
        crop_names = getattr(self.cfg, "TBL_PARTICLE_CROP_FILE_NAMES", {})
        template = crop_names.get(key)
        if template:
            return sample_dir / str(template).format(prefix=prefix)
        return sample_dir / f"{prefix}_{key}_crop.npy"

    def particle_image_path(self, sample_dir: Path, time_name: str, key: str) -> Path:
        """颗粒图像优先读 npy；npy 不存在时读取 evaluate/test_all 已经保存的 png。"""

        nested_name = {"lr": "lr.png", "gt": "hr.png", "sr": "fake.png"}[key]
        prefix = "prev" if time_name == "previous" else "next"
        flat_name = {"lr": f"{prefix}_lr.png", "gt": f"{prefix}_hr.png", "sr": f"{prefix}_sr.png"}[key]
        nested = sample_dir / time_name / nested_name
        return nested if nested.exists() else sample_dir / flat_name

    def particle_stats_csv_path(self, sample_dir: Path, time_name: str) -> Path:
        prefix = "prev" if time_name == "previous" else "next"
        nested = sample_dir / time_name / "particle_binary_stats_stats.csv"
        return nested if nested.exists() else sample_dir / f"{prefix}_particle_binary_stats_stats.csv"

    # =========================
    # 运行入口
    # =========================
    def run_all(self) -> None:
        self.run_started_at = time.perf_counter()
        self.timing_records = []
        groups = self.discover_groups()
        total_groups = len(groups)
        enabled_stages = self.enabled_output_stages()
        self.progress(f"discovered {total_groups} groups.")
        self.progress(f"enabled output stages: {', '.join(sorted(enabled_stages))}")
        for group_index, group in enumerate(groups, start=1):
            group_start = time.perf_counter()
            self.active_comparison_name = group.comparison_name
            self.progress(f"group {group_index}/{total_groups}: {self.group_progress_label(group)}")
            steps = []
            if "energy_spectrum" in enabled_stages:
                steps.append(("energy_spectrum", self.plot_energy_spectrum))
            if "error_histograms" in enabled_stages:
                steps.append(("histograms", self.plot_histogram_bundle))
            if "flow_u_epe_hist_overlay" in enabled_stages:
                steps.append(("flow_u_epe_hist_overlay", self.plot_flow_u_epe_histogram))
            if normalize_name(group.category_name) != "all":
                if "error_maps" in enabled_stages:
                    steps.append(("error_maps", self.plot_error_map_bundle))
                if "tbl_02_error_map" in enabled_stages and normalize_name(group.category_name) == "tbl":
                    steps.append(("tbl_02_error_map", self.plot_error_map_bundle))
                if "composite_panels" in enabled_stages:
                    steps.append(("composites", self.plot_composite_bundle))
                if "particle_stats_metrics" in enabled_stages:
                    steps.append(("particle_stats_metrics", self.plot_particle_stats_metric_only))
                if "composite_panels" in enabled_stages and normalize_name(group.category_name) in ("tbl", "twcf"):
                    steps.append(("tbl_twcf_flow_uv", self.plot_tbl_twcf_flow_uv))
                if (
                    ("composite_panels" in enabled_stages or "tbl_profile_overlay" in enabled_stages)
                    and normalize_name(group.category_name) == "tbl"
                ):
                    steps.append(("tbl_profile_overlay", self.plot_tbl_profile_overlays))
            step_total = len(steps)
            if step_total == 0:
                self.progress(f"group {group_index}/{total_groups} has no enabled steps: {self.group_progress_label(group)}")
                continue
            start_step_idx = self.resume_step_start_index(group_index, steps)
            if start_step_idx >= step_total:
                self.progress(f"group {group_index}/{total_groups} skipped by resume settings: {self.group_progress_label(group)}")
                continue
            if start_step_idx > 0:
                self.progress(
                    f"group {group_index}/{total_groups} resume from step "
                    f"{start_step_idx + 1}/{step_total}: {steps[start_step_idx][0]}"
                )
            for step_index, (step_name, step_func) in enumerate(steps[start_step_idx:], start=start_step_idx + 1):
                self.progress_step(group_index, total_groups, step_index, step_total, step_name)
                step_start = time.perf_counter()
                step_func(group)
                step_elapsed = time.perf_counter() - step_start
                self.timing_records.append(
                    {
                        "comparison": group.comparison_name,
                        "class": group.class_name,
                        "split": group.split_name,
                        "category": group.category_name,
                        "step": step_name,
                        "seconds": step_elapsed,
                        "duration": self.format_duration(step_elapsed),
                        "group_index": group_index,
                        "step_index": step_index,
                    }
                )
                self.progress_step_done(group_index, total_groups, step_index, step_total, step_name, step_elapsed)
            group_elapsed = time.perf_counter() - group_start
            group_done = f"group {group_index}/{total_groups} done: {self.group_progress_label(group)}"
            if getattr(self.cfg, "PROGRESS_SHOW_RUNTIME", True):
                group_done = f"{group_done} in {self.format_duration(group_elapsed)}"
            self.progress(group_done)
        self.progress("writing metric tables.")
        metric_start = time.perf_counter()
        self.write_metric_tables(groups)
        metric_elapsed = time.perf_counter() - metric_start
        self.timing_records.append(
            {"comparison": "all", "class": "all", "split": "all", "category": "all", "step": "metric_tables", "seconds": metric_elapsed, "duration": self.format_duration(metric_elapsed)}
        )
        self.progress("writing summary.")
        self.write_summary(groups)
        total_elapsed = time.perf_counter() - self.run_started_at if self.run_started_at is not None else 0.0
        self.progress(f"done in {self.format_duration(total_elapsed)}.")

    def write_summary(self, groups: list[GroupContext]) -> None:
        """写出本次扫描到的分组和缺文件信息，便于回头定位没有生成的图。"""

        # summary 放在 output/00_summary，记录两套对比组的扫描结果，不归入某一个组。
        summary_dir = self.output_root_dir / safe_name(self.cfg.SUMMARY_OUTPUT_DIR_NAME)
        summary_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "data_root": str(self.cfg.DATA_ROOT_DIR),
            "output_root": str(self.output_root_dir),
            "configured_output_root": str(self.cfg.OUTPUT_ROOT_DIR) if getattr(self.cfg, "OUTPUT_ROOT_DIR", None) is not None else None,
            "group_count": len(groups),
            "runtime_seconds": time.perf_counter() - self.run_started_at if self.run_started_at is not None else None,
            "runtime": self.format_duration(time.perf_counter() - self.run_started_at) if self.run_started_at is not None else None,
            "timing_records": self.timing_records,
            "groups": [
                {
                    "comparison_name": g.comparison_name,
                    "experiment_keys": list(g.experiment_keys),
                    "class_name": g.class_name,
                    "split_name": g.split_name,
                    "category_name": g.category_name,
                    "experiments": {k: str(v) for k, v in g.experiment_dirs.items()},
                }
                for g in groups
            ],
            "warnings": self.warnings,
        }
        (summary_dir / "summary.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # =========================
    # 指标汇总表
    # =========================
    def write_metric_tables(self, groups: list[GroupContext]) -> None:
        """把 ALL_CLASS 指标和 metrics_summary.csv 按类别汇总成 CSV 与带 sheet 的 xlsx。"""

        previous_comparison = self.active_comparison_name
        scopes: dict[tuple[str, str, str], list[GroupContext]] = {}
        for group in groups:
            # 指标表的 sheet 使用 backstep/cylinder 等具体类别；all 只是总目录分组，不作为 sheet。
            if normalize_name(group.category_name) == "all":
                continue
            scope_key = (group.comparison_name, group.class_name, group.split_name)
            scopes.setdefault(scope_key, []).append(group)

        for (comparison_name, class_name, split_name), scope_groups in sorted(scopes.items()):
            self.active_comparison_name = comparison_name
            out_dir = self.output_dir(self.cfg.METRIC_TABLE_OUTPUT_DIR_NAME, class_name, split_name)
            sheet_csv_dir = out_dir / safe_name(self.cfg.METRIC_TABLE_SHEET_CSV_DIR_NAME)
            sheet_csv_dir.mkdir(parents=True, exist_ok=True)
            category_tables: dict[str, list[dict[str, str]]] = {}
            flat_rows: list[dict[str, str]] = []
            summary_category_tables: dict[str, list[dict[str, str]]] = {}
            summary_flat_rows: list[dict[str, str]] = []
            summary_columns = list(getattr(self.cfg, "METRIC_SUMMARY_METADATA_COLUMNS", ()))

            for group in sorted(scope_groups, key=lambda item: normalize_name(item.category_name)):
                rows = self.build_metric_rows_for_group(group)
                if rows:
                    category_tables[group.category_name] = rows
                    flat_rows.extend(rows)
                    # CSV 本身不支持 sheet，因此每个类别额外输出一个单独 CSV，文件名对应 sheet 名。
                    self.write_csv_rows(
                        sheet_csv_dir / f"{safe_name(group.category_name)}.csv",
                        rows,
                        self.cfg.METRIC_TABLE_COLUMNS,
                    )

                # metrics_summary.csv 的列数和列名可能随实验版本变化，因此先收集所有类别的列名并保持首次出现顺序。
                summary_rows = self.build_metric_summary_rows_for_group(group)
                if summary_rows:
                    summary_category_tables[group.category_name] = summary_rows
                    summary_flat_rows.extend(summary_rows)
                    for row in summary_rows:
                        for column in row.keys():
                            if column not in summary_columns:
                                summary_columns.append(column)

            if flat_rows:
                self.write_csv_rows(
                    out_dir / self.cfg.METRIC_TABLE_FLAT_CSV_NAME,
                    flat_rows,
                    self.cfg.METRIC_TABLE_COLUMNS,
                )
                self.write_xlsx_workbook(
                    out_dir / self.cfg.METRIC_TABLE_WORKBOOK_NAME,
                    category_tables,
                    self.cfg.METRIC_TABLE_COLUMNS,
                )
            if summary_flat_rows:
                summary_columns_tuple = tuple(summary_columns)
                summary_sheet_csv_dir = out_dir / safe_name(self.cfg.METRIC_SUMMARY_SHEET_CSV_DIR_NAME)
                summary_sheet_csv_dir.mkdir(parents=True, exist_ok=True)
                for category_name, rows in summary_category_tables.items():
                    # metrics_summary 的每个类别也额外输出一个 CSV，等价于 xlsx 里的一个 sheet。
                    self.write_csv_rows(
                        summary_sheet_csv_dir / f"{safe_name(category_name)}.csv",
                        rows,
                        summary_columns_tuple,
                    )
                self.write_csv_rows(
                    out_dir / self.cfg.METRIC_SUMMARY_FLAT_CSV_NAME,
                    summary_flat_rows,
                    summary_columns_tuple,
                )
                self.write_xlsx_workbook(
                    out_dir / self.cfg.METRIC_SUMMARY_WORKBOOK_NAME,
                    summary_category_tables,
                    summary_columns_tuple,
                )

        self.active_comparison_name = previous_comparison

    def build_metric_rows_for_group(self, group: GroupContext) -> list[dict[str, str]]:
        """读取一个类别下各实验的 ALL_CLASS 指标，并按论文表格列名拼成多行。"""

        rows: list[dict[str, str]] = []
        for exp_key in [key for key in self.legend_order_keys() if key in group.experiment_dirs]:
            category_dir = group.experiment_dirs[exp_key]
            split_root = category_dir.parent if normalize_name(group.category_name) != "all" else category_dir
            flow_path = self.find_case_insensitive_file(
                split_root,
                ("ALL_CLASS_flow.csv", "ALL_CLASS_flow.CSV"),
            )
            image_path = self.find_case_insensitive_file(
                split_root,
                ("ALL_CLASS_IMAGE_PAIR.csv", "ALL_CLASS_IMAGE_PAIR.CSV"),
            )
            flow_row = self.find_category_metric_row(self.read_csv_dict_rows(flow_path), group.category_name)
            image_row = self.find_category_metric_row(self.read_csv_dict_rows(image_path), group.category_name)

            row = {
                "comparison": group.comparison_name,
                "class": group.class_name,
                "split": group.split_name,
                "category": group.category_name,
                "experiment": self.experiment_label(exp_key),
            }
            for out_key, aliases in self.cfg.METRIC_FLOW_FIELD_ALIASES.items():
                row[out_key] = self.metric_value_from_aliases(flow_row, aliases)
            for out_key, aliases in self.cfg.METRIC_IMAGE_FIELD_ALIASES.items():
                row[out_key] = self.metric_value_from_aliases(image_row, aliases)
            rows.append({column: row.get(column, "") for column in self.cfg.METRIC_TABLE_COLUMNS})

            if flow_path is None:
                self.warn(f"missing ALL_CLASS_flow.csv for {group.tag}/{exp_key}: {split_root}")
            if image_path is None:
                self.warn(f"missing ALL_CLASS_IMAGE_PAIR.csv for {group.tag}/{exp_key}: {split_root}")
        return rows

    def build_metric_summary_rows_for_group(self, group: GroupContext) -> list[dict[str, str]]:
        """读取 metrics_summary.csv，并把多行统计压缩成每个实验一行的对比记录。"""

        rows: list[dict[str, str]] = []
        file_names = tuple(getattr(self.cfg, "METRIC_SUMMARY_FILE_NAMES", ("metrics_summary.csv", "metrics_summary.CSV")))
        metadata_columns = tuple(getattr(self.cfg, "METRIC_SUMMARY_METADATA_COLUMNS", ()))
        for exp_key in [key for key in self.legend_order_keys() if key in group.experiment_dirs]:
            category_dir = group.experiment_dirs[exp_key]
            split_root = category_dir.parent if normalize_name(group.category_name) != "all" else category_dir
            # metrics_summary.csv 可能放在类别目录，也可能放在 split 根目录；
            # 优先使用类别目录，避免 split 根目录里的总表被每个类别重复读取。
            summary_path = self.find_case_insensitive_file(category_dir, file_names)
            if summary_path is None:
                summary_path = self.find_case_insensitive_file(split_root, file_names)
            csv_rows = self.read_csv_dict_rows(summary_path)
            selected_rows = self.filter_metric_summary_rows(csv_rows, group.category_name)
            if not selected_rows:
                selected_rows = csv_rows
            if not selected_rows:
                if summary_path is None:
                    self.warn(f"missing metrics_summary.csv for {group.tag}/{exp_key}: {category_dir}")
                continue

            summary_values = self.summarize_metric_summary_rows(selected_rows, metadata_columns)
            row = {
                "comparison": group.comparison_name,
                "class": group.class_name,
                "split": group.split_name,
                "category": group.category_name,
                "experiment": self.experiment_label(exp_key),
            }
            row.update(summary_values)
            rows.append(row)
        return rows

    def filter_metric_summary_rows(self, rows: list[dict[str, str]], category_name: str) -> list[dict[str, str]]:
        """如果 metrics_summary.csv 是 split 级总表，就按类别列筛出当前 backstep/cylinder 等类别。"""

        if not rows:
            return []
        target = normalize_name(category_name)
        if target == "all":
            return rows
        # 这些候选列名覆盖常见的类别/数据集/场景字段；找不到匹配列时返回空列表，
        # 调用方会回退到使用整个 metrics_summary.csv，兼容类别目录中只保存本类别数据的情况。
        category_keys = (
            "dataset",
            "category",
            "class",
            "class_name",
            "source_class",
            "data_type",
            "name",
            "case",
            "scene",
        )
        matched: list[dict[str, str]] = []
        for row in rows:
            normalized_keys = {normalize_name(key): key for key in row.keys()}
            for key in category_keys:
                real_key = normalized_keys.get(normalize_name(key))
                if real_key and normalize_name(row.get(real_key, "")) == target:
                    matched.append(row)
                    break
        return matched

    def summarize_metric_summary_rows(
        self,
        rows: list[dict[str, str]],
        reserved_columns: tuple[str, ...],
    ) -> dict[str, str]:
        """把 metrics_summary.csv 多行压缩为一行：前 11 列取首行，第 12 列起取每列最大值。"""

        if not rows:
            return {}
        fixed_count = int(getattr(self.cfg, "METRIC_SUMMARY_FIXED_COLUMN_COUNT", 11))
        source_columns = list(rows[0].keys())
        summary: dict[str, str] = {}
        used_columns = set(reserved_columns)
        for column_index, source_column in enumerate(source_columns):
            output_column = self.metric_summary_output_column_name(source_column, used_columns)
            used_columns.add(output_column)
            if column_index < fixed_count:
                # 前 11 列按用户要求“都是一行的”，直接取第一行；若第一行为空，取后续第一个非空值。
                summary[output_column] = self.first_nonempty_metric_summary_value(rows, source_column)
            else:
                # 后续每列在多行中取数值最大值；如果整列无法转成数值，则保留第一个非空文本。
                summary[output_column] = self.max_metric_summary_value(rows, source_column)
        return summary

    def metric_summary_output_column_name(self, source_column: str, used_columns: set[str]) -> str:
        """避免 metrics_summary 原始列名和 comparison/class 等定位列重名。"""

        base = str(source_column).strip() or "unnamed"
        # 如果原表也有 class/category/experiment 等列，添加 metrics_summary_ 前缀，避免覆盖定位列。
        candidate = base if base not in used_columns else f"metrics_summary_{base}"
        suffix = 2
        while candidate in used_columns:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def first_nonempty_metric_summary_value(self, rows: list[dict[str, str]], column: str) -> str:
        """取首行值；首行为空时向下寻找第一个非空值。"""

        first_value = str(rows[0].get(column, "")).strip()
        if first_value:
            return first_value
        for row in rows[1:]:
            text = str(row.get(column, "")).strip()
            if text:
                return text
        return ""

    def max_metric_summary_value(self, rows: list[dict[str, str]], column: str) -> str:
        """取 metrics_summary 某一列的最大数值；保留原始文本格式写入表格。"""

        best_number: float | None = None
        best_text = ""
        first_text = ""
        for row in rows:
            text = str(row.get(column, "")).strip()
            if text and not first_text:
                first_text = text
            number = self.to_float_loose(text)
            if number is None or not math.isfinite(number):
                continue
            if best_number is None or number > best_number:
                best_number = number
                best_text = text
        return best_text if best_number is not None else first_text

    def find_case_insensitive_file(self, directory: Path, names: tuple[str, ...]) -> Path | None:
        """按文件名大小写不敏感查找 CSV，兼容 .csv/.CSV 两种历史输出。"""

        for name in names:
            path = directory / name
            if path.exists():
                return path
        if not directory.exists():
            return None
        targets = {name.lower() for name in names}
        for child in directory.iterdir():
            if child.is_file() and child.name.lower() in targets:
                return child
        return None

    def read_csv_dict_rows(self, path: Path | None) -> list[dict[str, str]]:
        if path is None or not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                return list(csv.DictReader(file_obj))
        except Exception as exc:
            self.warn(f"failed to read metric csv {path}: {exc}")
            return []

    def find_category_metric_row(self, rows: list[dict[str, str]], category_name: str) -> dict[str, str]:
        """在 ALL_CLASS 表中找到当前类别的 CLASS_MEAN 行。"""

        if not rows:
            return {}
        target = normalize_name(category_name)
        category_keys = ("dataset", "category", "class", "class_name", "source_class", "data_type", "name")
        matched = []
        for row in rows:
            normalized_keys = {normalize_name(key): key for key in row.keys()}
            for key in category_keys:
                real_key = normalized_keys.get(normalize_name(key))
                if real_key and normalize_name(row.get(real_key, "")) == target:
                    matched.append(row)
                    break
        if not matched and len(rows) == 1:
            return rows[0]
        if not matched:
            return {}
        for row in matched:
            sample_index = row.get("sample_index", row.get("Sample_Index", ""))
            if normalize_name(sample_index) in ("class_mean", "mean", "all_mean"):
                return row
        return matched[0]

    def metric_value_from_aliases(self, row: dict[str, str], aliases: tuple[str, ...]) -> str:
        """按全局配置里的别名从 CSV 行中取值，并统一去掉空白。"""

        if not row:
            return ""
        normalized_keys = {normalize_name(key): key for key in row.keys()}
        for alias in aliases:
            if alias in row:
                return str(row.get(alias, "")).strip()
            real_key = normalized_keys.get(normalize_name(alias))
            if real_key:
                return str(row.get(real_key, "")).strip()
        return ""

    def write_csv_rows(self, path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)

    def write_xlsx_workbook(
        self,
        path: Path,
        category_tables: dict[str, list[dict[str, str]]],
        columns: tuple[str, ...],
    ) -> None:
        """不依赖 openpyxl，直接写一个最小 xlsx；每个类别对应一个 sheet。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        sheet_items = [(self.safe_sheet_name(name, idx), rows) for idx, (name, rows) in enumerate(category_tables.items(), start=1)]
        if not sheet_items:
            return
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(
                "[Content_Types].xml",
                self.xlsx_content_types(len(sheet_items)),
            )
            zip_file.writestr(
                "_rels/.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
            )
            zip_file.writestr("xl/workbook.xml", self.xlsx_workbook_xml([name for name, _ in sheet_items]))
            zip_file.writestr("xl/_rels/workbook.xml.rels", self.xlsx_workbook_rels_xml(len(sheet_items)))
            for idx, (_, rows) in enumerate(sheet_items, start=1):
                zip_file.writestr(f"xl/worksheets/sheet{idx}.xml", self.xlsx_sheet_xml(rows, columns))

    def safe_sheet_name(self, name: str, index: int) -> str:
        """Excel sheet 名最长 31 字符，且不能包含若干特殊符号。"""

        forbidden_chars = set("[]:*?/\\")
        cleaned = "".join("_" if char in forbidden_chars else char for char in str(name)).strip()
        if not cleaned:
            cleaned = f"Sheet{index}"
        return cleaned[:31]

    def xlsx_content_types(self, sheet_count: int) -> str:
        sheet_overrides = "\n".join(
            f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for idx in range(1, sheet_count + 1)
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{sheet_overrides}
</Types>"""

    def xlsx_workbook_xml(self, sheet_names: list[str]) -> str:
        sheets_xml = "\n".join(
            f'    <sheet name="{xml_escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
            for idx, name in enumerate(sheet_names, start=1)
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
{sheets_xml}
  </sheets>
</workbook>"""

    def xlsx_workbook_rels_xml(self, sheet_count: int) -> str:
        rels_xml = "\n".join(
            f'  <Relationship Id="rId{idx}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
            for idx in range(1, sheet_count + 1)
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels_xml}
</Relationships>"""

    def xlsx_sheet_xml(self, rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
        table_rows = [dict(zip(columns, columns))]
        table_rows.extend(rows)
        row_xml = []
        for row_idx, row in enumerate(table_rows, start=1):
            cell_xml = []
            for col_idx, column in enumerate(columns, start=1):
                ref = f"{self.xlsx_column_name(col_idx)}{row_idx}"
                cell_xml.append(self.xlsx_cell_xml(ref, row.get(column, "")))
            row_xml.append(f'    <row r="{row_idx}">{"".join(cell_xml)}</row>')
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
{chr(10).join(row_xml)}
  </sheetData>
</worksheet>"""

    def xlsx_column_name(self, index: int) -> str:
        name = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def xlsx_cell_xml(self, ref: str, value) -> str:
        if value is None or value == "":
            return f'<c r="{ref}"/>'
        text = str(value).strip()
        number = self.to_float(text)
        if number is not None and math.isfinite(number):
            return f'<c r="{ref}"><v>{xml_escape(text)}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{xml_escape(text)}</t></is></c>'

    # =========================
    # (1) 能量谱
    # =========================
    def plot_energy_spectrum(self, group: GroupContext) -> None:
        """把八个实验的 ENERGY_SPECTRUM 曲线叠加到一张图。"""

        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        gt_series: tuple[np.ndarray, np.ndarray] | None = None
        for exp_key, directory in group.experiment_dirs.items():
            # GT 能量谱在不同实验中理论一致，这里只读取第一条可用 GT 曲线并画成单独的 GT 图例。
            if gt_series is None:
                gt_path = first_existing(directory, self.cfg.ENERGY_SPECTRUM_GT_FILE_CANDIDATES)
                if gt_path is not None:
                    gt_xy = array_to_xy(load_npy(gt_path))
                    if gt_xy is not None:
                        gt_x, gt_y = gt_xy
                        gt_mask = np.isfinite(gt_x) & np.isfinite(gt_y) & (gt_x > 0) & (gt_y > 0)
                        if np.any(gt_mask):
                            gt_series = (gt_x[gt_mask], gt_y[gt_mask])

            path = first_existing(directory, self.cfg.ENERGY_SPECTRUM_FILE_CANDIDATES)
            if path is None:
                continue
            xy = array_to_xy(load_npy(path))
            if xy is None:
                self.warn(f"skip energy spectrum, cannot parse: {path}")
                continue
            x, y = xy
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(mask):
                series[exp_key] = (x[mask], y[mask])

        if not series and gt_series is None:
            return

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        if gt_series is not None:
            gt_x, gt_y = gt_series
            ax.loglog(
                gt_x,
                gt_y,
                label=self.cfg.GT_ENERGY_LABEL,
                color=self.cfg.GT_ENERGY_COLOR,
                linestyle=self.cfg.GT_ENERGY_LINESTYLE,
                linewidth=1.7,
            )
        for exp_key in self.legend_order_keys():
            if exp_key not in series:
                continue
            x, y = series[exp_key]
            ax.loglog(
                x,
                y,
                label=self.experiment_label(exp_key),
                color=self.experiment_color(exp_key),
                linewidth=1.6,
            )
        ax.set_xlabel(self.cfg.ENERGY_X_LABEL)
        ax.set_ylabel(self.cfg.ENERGY_Y_LABEL)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        self.apply_ordered_legend(ax, include_gt=True, energy_style=True)
        self.apply_axis_limits(
            ax,
            self.cfg.ENERGY_SPECTRUM_X_MIN,
            self.cfg.ENERGY_SPECTRUM_X_MAX,
            self.cfg.ENERGY_SPECTRUM_Y_MIN,
            self.cfg.ENERGY_SPECTRUM_Y_MAX,
        )

        out_dir = self.output_dir(self.cfg.ENERGY_OUTPUT_DIR_NAME, group.class_name, group.split_name)
        out_base = out_dir / f"{safe_name(group.category_name)}_energy_spectrum"
        self.save_npy(out_base.with_suffix(".npy"), {"gt": gt_series, "experiments": series})
        self.save_figure(fig, out_base)

    # =========================
    # (3)(4)(5) 直方图叠加
    # =========================
    def plot_histogram_bundle(self, group: GroupContext) -> None:
        """生成光流、颗粒、涡度误差直方图的跨实验叠加图。"""

        self.plot_overlay_histogram(
            group,
            file_candidates=("delta_w_hist_all.npy", "delta_w_hist.npy"),
            out_name="flow_error_hist_overlay",
            x_label=self.cfg.FLOW_ERROR_HIST_X_LABEL,
            y_label=self.cfg.HIST_Y_LABEL,
            axis_kind="flow",
            save_npy=True,
        )
        self.plot_flow_u_epe_histogram(group)
        self.plot_overlay_histogram(
            group,
            file_candidates=(self.cfg.PARTICLE_HIST_FILE_NAME, self.cfg.PARTICLE_SAMPLE_HIST_FILE_NAME),
            out_name="particle_error_hist_overlay",
            x_label=self.cfg.PARTICLE_ERROR_HIST_X_LABEL,
            y_label=self.cfg.HIST_Y_LABEL,
            axis_kind="particle",
            save_npy=False,
        )
        self.plot_overlay_histogram(
            group,
            file_candidates=(self.cfg.VORTICITY_HIST_FILE_NAME, self.cfg.VORTICITY_SAMPLE_HIST_FILE_NAME),
            out_name="vorticity_error_hist_overlay",
            x_label=self.cfg.VORTICITY_ERROR_HIST_X_LABEL,
            y_label=self.cfg.HIST_Y_LABEL,
            axis_kind="vorticity",
            save_npy=False,
        )

    def load_hist_series(
        self, group: GroupContext, file_candidates: tuple[str, ...]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """读取每个实验的直方图 npy，并统一转换成中心点/计数。"""

        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for exp_key, directory in group.experiment_dirs.items():
            path = first_existing(directory, file_candidates)
            if path is None:
                continue
            xy = array_to_xy(load_npy(path))
            if xy is None:
                self.warn(f"skip histogram, cannot parse: {path}")
                continue
            x, y = xy
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                series[exp_key] = (x[mask], y[mask])
        return series

    def plot_overlay_histogram(
        self,
        group: GroupContext,
        file_candidates: tuple[str, ...],
        out_name: str,
        x_label: str,
        y_label: str,
        axis_kind: str,
        save_npy: bool,
    ) -> None:
        series = self.load_hist_series(group, file_candidates)
        if not series:
            return

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for draw_idx, (exp_key, (x, y)) in enumerate(self.sorted_hist_series(series)):
            width = self.estimate_bar_width(x)
            hist_color = self.experiment_hist_color(exp_key)
            # 误差直方图按参考图只使用半透明填充，不再加粗柱边框；如需恢复轮廓线可在 global_class.py 开启。
            edge_width = float(getattr(self.cfg, "HIST_EDGE_LINE_WIDTH", 0.0))
            edge_color = "none" if edge_width <= 0 else self.darken_color(hist_color)
            ax.bar(
                x,
                y,
                width=width,
                color=hist_color,
                alpha=self.cfg.HIST_ALPHA,
                edgecolor=edge_color,
                linewidth=edge_width,
                label=self.experiment_label(exp_key),
                zorder=self.hist_zorder(exp_key, draw_idx),
            )
            if getattr(self.cfg, "HIST_DRAW_OUTLINE", False) and float(getattr(self.cfg, "HIST_LINE_WIDTH", 0.0)) > 0:
                ax.plot(
                    x,
                    y,
                    color=self.darken_color(hist_color),
                    linewidth=self.cfg.HIST_LINE_WIDTH,
                    alpha=0.95,
                    zorder=self.hist_zorder(exp_key, draw_idx) + 0.1,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        self.apply_hist_legend(ax, series)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        self.apply_hist_axis(ax, group.category_name, axis_kind)

        out_dir = self.output_dir(self.cfg.HIST_OUTPUT_DIR_NAME, group.class_name, group.split_name)
        out_base = out_dir / f"{safe_name(group.category_name)}_{out_name}"
        if save_npy:
            self.save_npy(out_base.with_suffix(".npy"), series)
        self.save_figure(fig, out_base)

    def plot_flow_u_epe_histogram(self, group: GroupContext) -> None:
        """左侧叠加 Δu 误差直方图，右侧叠加 EPE 直方图。"""

        u_series = self.load_hist_series(group, ("delta_u_hist_all.npy", "delta_u_hist.npy"))
        epe_series = self.load_hist_series(group, ("epe_hist_all.npy", "epe_hist.npy"))
        if not u_series and not epe_series:
            return

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.4, 3.4),
            gridspec_kw={"wspace": float(getattr(self.cfg, "FLOW_U_EPE_HIST_WSPACE", 0.32))},
        )
        for ax, series, x_label, axis_kind in (
            (axes[0], u_series, self.cfg.FLOW_U_HIST_X_LABEL, "flow_u"),
            (axes[1], epe_series, self.cfg.EPE_HIST_X_LABEL, "epe"),
        ):
            for draw_idx, (exp_key, (x, y)) in enumerate(self.sorted_hist_series(series)):
                hist_color = self.experiment_hist_color(exp_key)
                edge_width = float(getattr(self.cfg, "HIST_EDGE_LINE_WIDTH", 0.0))
                edge_color = "none" if edge_width <= 0 else self.darken_color(hist_color)
                ax.bar(
                    x,
                    y,
                    width=self.estimate_bar_width(x),
                    color=hist_color,
                    alpha=self.cfg.HIST_ALPHA,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    label=self.experiment_label(exp_key),
                    zorder=self.hist_zorder(exp_key, draw_idx),
                )
                if getattr(self.cfg, "HIST_DRAW_OUTLINE", False) and float(getattr(self.cfg, "HIST_LINE_WIDTH", 0.0)) > 0:
                    ax.plot(
                        x,
                        y,
                        color=self.darken_color(hist_color),
                        linewidth=self.cfg.HIST_LINE_WIDTH,
                        zorder=self.hist_zorder(exp_key, draw_idx) + 0.1,
                    )
            ax.set_xlabel(x_label)
            ax.set_ylabel(self.cfg.HIST_Y_LABEL)
            ax.grid(True, alpha=0.18, linewidth=0.5)
            self.apply_hist_axis(ax, group.category_name, axis_kind)
        self.apply_hist_legend(axes[1], epe_series or u_series)

        out_dir = self.output_dir(self.cfg.HIST_OUTPUT_DIR_NAME, group.class_name, group.split_name)
        out_base = out_dir / f"{safe_name(group.category_name)}_flow_u_epe_hist_overlay"
        self.save_figure(fig, out_base)

    def sorted_hist_series(
        self, series: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
        """按分布宽高排序；指定顶层实验会被强制移到最后绘制，避免被覆盖。"""

        def score(item: tuple[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
            _, (x, y) = item
            positive = x[y > 0]
            width = float(np.nanmax(positive) - np.nanmin(positive)) if positive.size else 0.0
            height = float(np.nanmax(y)) if y.size else 0.0
            return width, height

        ordered = sorted(series.items(), key=score, reverse=True)
        top_keys = tuple(getattr(self.cfg, "HIST_TOP_EXPERIMENT_KEYS", ()))
        top_set = set(top_keys)
        normal_items = [item for item in ordered if item[0] not in top_set]
        # 顶层实验按全局指定顺序追加到最后，绘制顺序和 zorder 都保证它们在最上层。
        top_items = [item for key in top_keys for item in ordered if item[0] == key]
        return normal_items + top_items

    def hist_zorder(self, exp_key: str, draw_idx: int) -> float:
        """误差直方图层级：ESRuRAFT-PIV 等指定实验始终高于其它实验。"""

        top_keys = tuple(getattr(self.cfg, "HIST_TOP_EXPERIMENT_KEYS", ()))
        if exp_key in top_keys:
            return 100.0 + float(top_keys.index(exp_key))
        return 10.0 + float(draw_idx)

    def estimate_bar_width(self, x: np.ndarray) -> float:
        finite_x = np.asarray(x, dtype=np.float64)
        if finite_x.size < 2:
            return 0.8
        diffs = np.diff(np.sort(finite_x))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        return float(np.median(diffs)) if diffs.size else 0.8

    def apply_axis_limits(
        self,
        ax: plt.Axes,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
    ) -> None:
        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

    def apply_hist_axis(self, ax: plt.Axes, category_name: str, axis_kind: str) -> None:
        """按普通类别或 TBL/TWCF 专用配置设置直方图坐标轴。"""

        category_norm = normalize_name(category_name)
        if axis_kind in ("flow", "flow_u"):
            limits = self.cfg.FLOW_ERROR_HIST_CATEGORY_AXIS_LIMITS.get(category_norm)
            x_min = self.cfg.FLOW_ERROR_HIST_X_MIN if axis_kind == "flow" else self.cfg.FLOW_U_HIST_X_MIN
            x_max = self.cfg.FLOW_ERROR_HIST_X_MAX if axis_kind == "flow" else self.cfg.FLOW_U_HIST_X_MAX
            y_min = self.cfg.FLOW_ERROR_HIST_Y_MIN if axis_kind == "flow" else self.cfg.FLOW_U_HIST_Y_MIN
            y_max = self.cfg.FLOW_ERROR_HIST_Y_MAX if axis_kind == "flow" else self.cfg.FLOW_U_HIST_Y_MAX
        elif axis_kind == "epe":
            limits = self.cfg.FLOW_ERROR_HIST_CATEGORY_AXIS_LIMITS.get(category_norm)
            x_min, x_max = self.cfg.EPE_HIST_X_MIN, self.cfg.EPE_HIST_X_MAX
            y_min, y_max = self.cfg.EPE_HIST_Y_MIN, self.cfg.EPE_HIST_Y_MAX
        elif axis_kind == "particle":
            limits = self.cfg.PARTICLE_ERROR_HIST_CATEGORY_AXIS_LIMITS.get(category_norm)
            x_min, x_max = self.cfg.PARTICLE_ERROR_HIST_X_MIN, self.cfg.PARTICLE_ERROR_HIST_X_MAX
            y_min, y_max = self.cfg.PARTICLE_ERROR_HIST_Y_MIN, self.cfg.PARTICLE_ERROR_HIST_Y_MAX
        else:
            limits = self.cfg.VORTICITY_ERROR_HIST_CATEGORY_AXIS_LIMITS.get(category_norm)
            x_min, x_max = self.cfg.VORTICITY_ERROR_HIST_X_MIN, self.cfg.VORTICITY_ERROR_HIST_X_MAX
            y_min, y_max = self.cfg.VORTICITY_ERROR_HIST_Y_MIN, self.cfg.VORTICITY_ERROR_HIST_Y_MAX

        if limits:
            x_min = limits.get("x_min", x_min)
            x_max = limits.get("x_max", x_max)
            y_min = limits.get("y_min", y_min)
            y_max = limits.get("y_max", y_max)
        self.apply_axis_limits(ax, x_min, x_max, y_min, y_max)

    # =========================
    # (2) 误差图统一色条
    # =========================
    def plot_error_map_bundle(self, group: GroupContext) -> None:
        self.plot_flow_error_maps(group)
        self.plot_particle_error_maps(group)

    def error_colorbar_reference_arrays(self, maps: dict[str, np.ndarray]) -> list[np.ndarray]:
        """光流/颗粒误差色条只按全局指定实验取范围，不再被所有实验的极端值撑大。"""

        group_refs = getattr(self.cfg, "COMPARISON_GROUP_ERROR_COLORBAR_REFERENCE_KEYS", {}).get(
            self.active_comparison_name
        )
        reference_keys = group_refs or getattr(self.cfg, "ERROR_COLORBAR_REFERENCE_EXPERIMENT_KEYS", ())
        refs = [maps[key] for key in reference_keys if key in maps and maps[key] is not None]
        if refs:
            return refs
        # 如果某个类别缺少参考实验，回退到当前已有实验，避免因为缺文件导致整张图无法生成。
        return [array for array in maps.values() if array is not None]

    def plot_flow_error_maps(self, group: GroupContext) -> None:
        """为每个样本生成 u/v/s 光流误差图，并按指定参考实验统一色条。"""

        for bundle in self.bundle_samples(group, "flow"):
            for component in ("u", "v", "s"):
                maps: dict[str, np.ndarray] = {}
                for exp_key, sample_dir in bundle.sample_dirs.items():
                    error_maps = self.load_flow_error_maps(sample_dir)
                    if component in error_maps:
                        maps[exp_key] = error_maps[component]
                if not maps:
                    continue
                # 光流误差图必须让 0 位于色条中心且显示为白色，因此范围强制关于 0 对称。
                vmin, vmax = self.resolve_color_limit(
                    self.error_colorbar_reference_arrays(maps),
                    self.cfg.FLOW_ERROR_COLORBAR_LIMIT,
                    center_zero=True,
                )
                out_dir = self.output_dir(
                    self.cfg.ERROR_MAP_OUTPUT_DIR_NAME,
                    group.class_name,
                    group.split_name,
                    group.category_name,
                    bundle.sample_name,
                )
                out_base = out_dir / f"flow_{component}_error"
                self.save_npy(out_base.with_suffix(".npy"), maps)
                self.plot_single_row_maps(
                    maps,
                    out_base,
                    cmap=self.cfg.ERROR_CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_label=self.cfg.FLOW_ERROR_COLORBAR_LABEL,
                )

    def plot_particle_error_maps(self, group: GroupContext) -> None:
        """为 previous/next 颗粒误差图统一色条范围并保存 png/svg。"""

        for bundle in self.bundle_samples(group, "particle"):
            is_tbl = normalize_name(group.category_name) == "tbl"
            for time_name in ("previous", "next"):
                if is_tbl and getattr(self.cfg, "TBL_ERROR_MAP_PARTICLE_PAIR_LAYOUT", True):
                    self.plot_tbl_particle_error_pair_map(group, bundle, time_name, crop=False)
                    if getattr(self.cfg, "TBL_PARTICLE_CROP_ENABLED", True):
                        self.plot_tbl_particle_error_pair_map(group, bundle, time_name, crop=True)
                    continue
                maps: dict[str, np.ndarray] = {}
                for exp_key, sample_dir in bundle.sample_dirs.items():
                    error = self.load_particle_array(sample_dir, time_name, "error")
                    if error is not None:
                        maps[exp_key] = ensure_2d_image(error)
                if not maps:
                    continue
                # 颗粒误差图同样强制关于 0 对称，保证正负误差颜色对等、0 为白色。
                vmin, vmax = self.resolve_color_limit(
                    self.error_colorbar_reference_arrays(maps),
                    self.cfg.PARTICLE_ERROR_COLORBAR_LIMIT,
                    center_zero=True,
                )
                out_dir = self.output_dir(
                    self.cfg.ERROR_MAP_OUTPUT_DIR_NAME,
                    group.class_name,
                    group.split_name,
                    group.category_name,
                    bundle.sample_name,
                )
                out_base = out_dir / f"particle_{time_name}_error"
                self.plot_single_row_maps(
                    maps,
                    out_base,
                    cmap=self.cfg.ERROR_CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_label=self.cfg.PARTICLE_ERROR_COLORBAR_LABEL,
                )

    def plot_tbl_particle_error_pair_map(
        self, group: GroupContext, bundle: SampleBundle, time_name: str, crop: bool = False
    ) -> None:
        """TBL 专用 02_error_maps：full-frame 首行加入 GT，并用红框标出 crop 位置。"""

        exp_keys = [key for key in self.legend_order_keys() if key in bundle.sample_dirs]
        rows: list[tuple[str, np.ndarray | None, np.ndarray | None]] = []
        sr_maps: dict[str, np.ndarray] = {}
        error_maps: dict[str, np.ndarray] = {}
        gt = None if crop else self.first_available_particle(bundle, time_name, "gt", crop=False)
        if gt is not None:
            gt = ensure_2d_image(gt)
            rows.append((self.cfg.GT_PANEL_LABEL, gt, None))
        for exp_key in exp_keys:
            sample_dir = bundle.sample_dirs.get(exp_key)
            if sample_dir is None:
                continue
            sr = self.load_particle_array_mode(sample_dir, time_name, "sr", crop=crop)
            error = self.load_particle_array_mode(sample_dir, time_name, "error", crop=crop)
            if sr is not None:
                sr_maps[exp_key] = ensure_2d_image(sr)
            if error is not None:
                error_maps[exp_key] = ensure_2d_image(error)
            if sr is not None or error is not None:
                rows.append((self.experiment_label(exp_key), sr, error))
        if not rows:
            return

        image_arrays_for_limit = ([gt] if gt is not None else []) + list(sr_maps.values())
        image_vmin, image_vmax = self.row_limit(image_arrays_for_limit, self.cfg.PARTICLE_VALUE_COLORBAR_LIMIT)
        error_vmin, error_vmax = self.resolve_color_limit(
            self.error_colorbar_reference_arrays(error_maps),
            self.cfg.PARTICLE_ERROR_COLORBAR_LIMIT,
            center_zero=True,
        )
        crop_ref = gt if gt is not None else next((value for value in sr_maps.values() if value is not None), None)
        crop_ref_hw = self.image_hw(crop_ref)
        crop_bounds = self.resolve_tbl_particle_crop_bounds(crop_ref_hw) if not crop else None
        row_count = len(rows)
        fig_width = float(getattr(self.cfg, "TBL_ERROR_MAP_VERTICAL_FIG_WIDTH", 10.5))
        row_height = float(getattr(self.cfg, "TBL_ERROR_MAP_VERTICAL_ROW_HEIGHT", 1.15))
        # 用户指定 TBL full-frame 的 particle_*_error.png 去掉第二列误差图和全部色条；
        # crop 图仍保留原来的两列布局，便于查看局部误差细节。
        full_frame_image_only = (
            not crop
            and normalize_name(group.category_name) == "tbl"
            and bool(getattr(self.cfg, "TBL_ERROR_MAP_PARTICLE_FULL_FRAME_IMAGE_ONLY", True))
        )
        label_loc = getattr(self.cfg, "TBL_ERROR_MAP_PARTICLE_LABEL_LOC", "upper_right") if full_frame_image_only else "upper_left"
        fig = plt.figure(figsize=(fig_width, max(3.0, row_height * row_count)))
        if full_frame_image_only:
            gs = fig.add_gridspec(
                row_count,
                1,
                hspace=float(getattr(self.cfg, "TBL_ERROR_MAP_VERTICAL_HSPACE", 0.06)),
                wspace=0.0,
            )
        else:
            gs = fig.add_gridspec(
                row_count,
                4,
                width_ratios=[1, 1, 0.045, 0.045],
                hspace=float(getattr(self.cfg, "TBL_ERROR_MAP_VERTICAL_HSPACE", 0.06)),
                wspace=float(getattr(self.cfg, "TBL_ERROR_MAP_VERTICAL_WSPACE", 0.05)),
            )
        image_handle = None
        error_handle = None
        for row_idx, (label, sr, error) in enumerate(rows):
            ax = fig.add_subplot(gs[row_idx, 0] if not full_frame_image_only else gs[row_idx])
            image_handle = self.draw_map(
                ax,
                sr,
                self.cfg.IMAGE_CMAP,
                image_vmin,
                image_vmax,
                label,
                fill_panel=crop,
                label_loc=label_loc,
            ) or image_handle
            self.draw_tbl_particle_crop_box(ax, sr, crop_ref_hw, crop_bounds)
            if not full_frame_image_only:
                ax = fig.add_subplot(gs[row_idx, 1])
                error_handle = self.draw_map(
                    ax,
                    error,
                    self.cfg.ERROR_CMAP,
                    error_vmin,
                    error_vmax,
                    label if error is not None else "",
                    fill_panel=crop,
                ) or error_handle
                self.draw_tbl_particle_crop_box(ax, error, crop_ref_hw, crop_bounds)

        if not full_frame_image_only:
            cax_image = fig.add_subplot(gs[:, 2])
            if image_handle is not None and image_vmin is not None and image_vmax is not None:
                cb = fig.colorbar(image_handle, cax=cax_image)
                cb.set_label(self.cfg.PARTICLE_VALUE_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
            else:
                cax_image.axis("off")
            cax_error = fig.add_subplot(gs[:, 3])
            if error_handle is not None and error_vmin is not None and error_vmax is not None:
                cb = fig.colorbar(error_handle, cax=cax_error)
                cb.set_label(self.cfg.PARTICLE_ERROR_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
            else:
                cax_error.axis("off")

        out_dir = self.output_dir(
            self.cfg.ERROR_MAP_OUTPUT_DIR_NAME,
            group.class_name,
            group.split_name,
            group.category_name,
            bundle.sample_name,
        )
        suffix = getattr(self.cfg, "TBL_PARTICLE_CROP_OUTPUT_SUFFIX", "_crop") if crop else ""
        out_base = out_dir / f"particle_{time_name}_error{suffix}"
        self.save_npy(out_base.with_suffix(".npy"), {"gt": gt, "sr": sr_maps, "error": error_maps})
        self.save_figure(fig, out_base)

    def plot_single_row_maps(
        self,
        maps: dict[str, np.ndarray],
        out_base: Path,
        cmap: str,
        vmin: float,
        vmax: float,
        colorbar_label: str,
    ) -> None:
        """把多个实验的同类误差图横向排布，末尾只放一个统一色条。"""

        exp_keys = [key for key in self.legend_order_keys() if key in maps]
        if not exp_keys:
            return
        fig = plt.figure(figsize=(2.2 * len(exp_keys) + 0.45, 2.2))
        gs = fig.add_gridspec(1, len(exp_keys) + 1, width_ratios=[1] * len(exp_keys) + [0.06])
        image_handle = None
        for idx, exp_key in enumerate(exp_keys):
            ax = fig.add_subplot(gs[0, idx])
            image_handle = ax.imshow(maps[exp_key], cmap=cmap, vmin=vmin, vmax=vmax)
            self.panel_text(ax, self.experiment_label(exp_key))
            ax.axis("off")
        cax = fig.add_subplot(gs[0, len(exp_keys)])
        if image_handle is not None:
            cb = fig.colorbar(image_handle, cax=cax)
            cb.set_label(colorbar_label, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
        self.save_figure(fig, out_base)

    def resolve_color_limit(
        self,
        arrays: Iterable[np.ndarray],
        limit: str | tuple[float, float] | list[float],
        center_zero: bool = False,
    ) -> tuple[float, float]:
        """按照全局配置解析色条范围；auto 时读取当前图组的真实 min/max。"""

        if isinstance(limit, (tuple, list)) and len(limit) == 2:
            vmin, vmax = float(limit[0]), float(limit[1])
            if center_zero:
                max_abs = max(abs(vmin), abs(vmax), 1e-6)
                return -max_abs, max_abs
            return vmin, vmax
        values = []
        for array in arrays:
            finite = finite_values(np.asarray(array))
            if finite.size:
                values.append(finite)
        if not values:
            return -1.0, 1.0
        merged = np.concatenate(values)
        if center_zero:
            max_abs = float(np.max(np.abs(merged))) if merged.size else 1.0
            max_abs = max(max_abs, 1e-6)
            return -max_abs, max_abs
        vmin, vmax = float(np.min(merged)), float(np.max(merged))
        if math.isclose(vmin, vmax):
            pad = abs(vmin) * 0.05 + 1e-6
            return vmin - pad, vmax + pad
        return vmin, vmax

    # =========================
    # 数据读取：光流 / 颗粒 / 涡度
    # =========================
    def load_flow_pair(self, sample_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        pred_path = sample_dir / self.cfg.FLOW_ARRAY_FILE_NAMES["pred"]
        gt_path = sample_dir / self.cfg.FLOW_ARRAY_FILE_NAMES["gt"]
        pred = flow_to_hw2(load_npy(pred_path)) if pred_path.exists() else None
        gt = flow_to_hw2(load_npy(gt_path)) if gt_path.exists() else None
        return pred, gt

    def load_flow_error_maps(self, sample_dir: Path) -> dict[str, np.ndarray]:
        """优先由 fake_flo/hr_flo 计算误差；缺失时读取 test_all 已保存的 delta_u/delta_v/delta_s。"""

        pred, gt = self.load_flow_pair(sample_dir)
        maps: dict[str, np.ndarray] = {}
        if pred is not None and gt is not None:
            common_h = min(pred.shape[0], gt.shape[0])
            common_w = min(pred.shape[1], gt.shape[1])
            pred = pred[:common_h, :common_w, :]
            gt = gt[:common_h, :common_w, :]
            maps["u"] = pred[..., 0] - gt[..., 0]
            maps["v"] = pred[..., 1] - gt[..., 1]
            maps["s"] = np.linalg.norm(pred, axis=-1) - np.linalg.norm(gt, axis=-1)
            return maps

        for component, file_name in (("u", "delta_u.npy"), ("v", "delta_v.npy"), ("s", "delta_s.npy")):
            array = load_npy(sample_dir / file_name)
            if array is not None:
                maps[component] = np.squeeze(array).astype(np.float32)

        delta_uvw = load_npy(sample_dir / "delta_uvw.npy")
        if delta_uvw is not None:
            arr = flow_to_hw2(delta_uvw)
            if arr is not None:
                maps.setdefault("u", arr[..., 0])
                maps.setdefault("v", arr[..., 1])
        return maps

    def widest_contiguous_region(self, indices: np.ndarray) -> tuple[int, int] | None:
        """从一组连续/非连续下标中找出最长连续段，用于从渲染图里定位真正的图像面板。"""

        if indices.size == 0:
            return None
        start = previous = int(indices[0])
        best = (start, previous + 1)
        for value in indices[1:]:
            current = int(value)
            if current == previous + 1:
                previous = current
                continue
            if previous + 1 - start > best[1] - best[0]:
                best = (start, previous + 1)
            start = previous = current
        if previous + 1 - start > best[1] - best[0]:
            best = (start, previous + 1)
        return best

    def crop_uvs_compare_panel(
        self,
        image: np.ndarray,
        row_idx: int,
        col_idx: int,
        row_count: int,
        col_count: int,
    ) -> np.ndarray | None:
        """
        从 test_all 已保存的 uvs_compare.png 中裁出单个 Pred/GT 面板。
        有些 test_all 只保存 uvs_compare.png 与 delta_*.npy，没有 fake_flo/hr_flo.npy；
        如果不从 PNG 回退读取，组合图里的 GT U/V/S 就会是空白。这里通过“非白色像素占比”
        自动定位彩色面板，并排除标题文字和右侧 colorbar。
        """

        height, width = image.shape[:2]
        if height <= 0 or width <= 0 or row_count <= 0 or col_count <= 0:
            return None

        y0 = int(round(row_idx * height / row_count))
        y1 = int(round((row_idx + 1) * height / row_count))
        x0 = int(round(col_idx * width / col_count))
        x1 = int(round((col_idx + 1) * width / col_count))
        cell = image[y0:y1, x0:x1]
        if cell.size == 0 or cell.ndim != 3:
            return None

        threshold = float(getattr(self.cfg, "FLOW_UVS_COMPARE_WHITE_THRESHOLD", 0.985))
        row_fraction_limit = float(getattr(self.cfg, "FLOW_UVS_COMPARE_ROW_MASK_FRACTION", 0.25))
        col_fraction_limit = float(getattr(self.cfg, "FLOW_UVS_COMPARE_COL_MASK_FRACTION", 0.35))
        min_fraction = float(getattr(self.cfg, "FLOW_UVS_COMPARE_MIN_PANEL_FRACTION", 0.25))

        # 渲染图背景接近纯白，真正的数据面板和 colorbar 都会有大量非白像素；
        # 先按行找出数据面板所在的纵向区间，再按列找出最宽的非白色连续块，窄 colorbar 会被自然排除。
        mask = np.any(cell[..., :3] < threshold, axis=2)
        row_indices = np.flatnonzero(mask.mean(axis=1) > row_fraction_limit)
        if row_indices.size:
            local_y0 = int(row_indices[0])
            local_y1 = int(row_indices[-1] + 1)
            col_indices = np.flatnonzero(mask[local_y0:local_y1].mean(axis=0) > col_fraction_limit)
            col_segment = self.widest_contiguous_region(col_indices)
            if col_segment is not None:
                local_x0, local_x1 = col_segment
                if (
                    local_y1 - local_y0 >= min_fraction * cell.shape[0]
                    and local_x1 - local_x0 >= min_fraction * cell.shape[1]
                ):
                    return np.ascontiguousarray(cell[local_y0:local_y1, local_x0:local_x1, :3])

        # 如果某张图颜色很浅导致自动定位失败，则使用全局给出的保守比例裁剪；
        # 这些比例只作为兜底，不影响正常情况下的自动裁剪。
        ratios = getattr(self.cfg, "FLOW_UVS_COMPARE_FALLBACK_CROP_RATIOS", {})
        left = float(ratios.get("left", 0.02))
        right = float(ratios.get("right", 0.82))
        top = float(ratios.get("top", 0.09))
        bottom = float(ratios.get("bottom", 0.96))
        local_x0 = max(0, min(cell.shape[1] - 1, int(round(left * cell.shape[1]))))
        local_x1 = max(local_x0 + 1, min(cell.shape[1], int(round(right * cell.shape[1]))))
        local_y0 = max(0, min(cell.shape[0] - 1, int(round(top * cell.shape[0]))))
        local_y1 = max(local_y0 + 1, min(cell.shape[0], int(round(bottom * cell.shape[0]))))
        return np.ascontiguousarray(cell[local_y0:local_y1, local_x0:local_x1, :3])

    def load_uvs_compare_value_maps(self, sample_dir: Path) -> dict[str, np.ndarray]:
        """从 uvs_compare.png 回退读取 Pred/GT 的 U/V/S 面板，用于缺少原始光流 npy 的 test_all。"""

        image = load_png(sample_dir / getattr(self.cfg, "FLOW_UVS_COMPARE_FILE_NAME", "uvs_compare.png"))
        if image is None:
            return {}
        arr = ensure_2d_image(image)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return {}
        rgb = np.asarray(arr[..., :3], dtype=np.float32)
        if np.nanmax(rgb) > 1.5:
            rgb = rgb / 255.0

        components = tuple(getattr(self.cfg, "FLOW_UVS_COMPARE_ROW_COMPONENTS", ("u", "v", "s")))
        value_columns = dict(getattr(self.cfg, "FLOW_UVS_COMPARE_VALUE_COLUMNS", {"pred": 0, "gt": 1}))
        col_count = int(getattr(self.cfg, "FLOW_UVS_COMPARE_TOTAL_COLUMNS", 3))
        maps: dict[str, np.ndarray] = {}
        for row_idx, component in enumerate(components):
            for value_kind, col_idx in value_columns.items():
                crop = self.crop_uvs_compare_panel(rgb, row_idx, int(col_idx), len(components), col_count)
                if crop is not None:
                    maps[f"{value_kind}_{component}"] = crop
        return maps

    def load_flow_value_maps(self, sample_dir: Path) -> dict[str, np.ndarray]:
        """读取预测/GT 光流的 u/v/s；缺少原始 npy 时优先从 uvs_compare.png 回退裁剪。"""

        pred, gt = self.load_flow_pair(sample_dir)
        maps: dict[str, np.ndarray] = {}
        if pred is not None:
            maps["pred_u"] = pred[..., 0]
            maps["pred_v"] = pred[..., 1]
            maps["pred_s"] = np.linalg.norm(pred, axis=-1)
        if gt is not None:
            maps["gt_u"] = gt[..., 0]
            maps["gt_v"] = gt[..., 1]
            maps["gt_s"] = np.linalg.norm(gt, axis=-1)
        # test_all 常见情况：没有 fake_flo/hr_flo.npy，但 uvs_compare.png 已经包含 Pred 与 GT。
        # 这里用 PNG 裁剪结果补齐缺失键，尤其修复组合图里的 GT U/V/S 空白问题。
        for key, value in self.load_uvs_compare_value_maps(sample_dir).items():
            maps.setdefault(key, value)
        if maps:
            return maps

        # 最后的兜底：少数历史目录可能连 uvs_compare.png 也没有，只保存 delta_u/delta_v；
        # 这类数据无法还原真实 Pred/GT，只能用 delta 图占位，保证旧结果目录仍能生成面板而不中断。
        errors = self.load_flow_error_maps(sample_dir)
        for key, value in errors.items():
            maps[f"pred_{key}"] = value
        return maps

    def load_particle_array(self, sample_dir: Path, time_name: str, key: str) -> np.ndarray | None:
        path = self.particle_path(sample_dir, time_name, key)
        array = load_npy(path)
        if array is not None:
            return array
        if key in ("lr", "gt", "sr"):
            return load_png(self.particle_image_path(sample_dir, time_name, key))
        return None

    def resolve_tbl_particle_crop_bounds(
        self, image_hw: tuple[int, int] | None
    ) -> tuple[int, int, int, int] | None:
        """按原始 evaluate 逻辑计算 TBL 颗粒 crop 框：默认 256x256，x 中心为宽度的 0.265。"""

        if image_hw is None:
            return None
        image_h, image_w = image_hw
        if image_h <= 0 or image_w <= 0:
            return None
        crop_size = int(getattr(self.cfg, "TBL_PARTICLE_CROP_SIZE", 256))
        center_ratio = float(getattr(self.cfg, "TBL_PARTICLE_CROP_CENTER_RATIO", 0.265))
        crop_h = int(np.clip(crop_size, 1, image_h))
        crop_w = int(np.clip(crop_size, 1, image_w))
        center_x = int(round(image_w * center_ratio))
        half_w = crop_w // 2
        x_start = center_x - half_w
        x_end = x_start + crop_w
        if x_start < 0:
            x_start = 0
            x_end = crop_w
        if x_end > image_w:
            x_end = image_w
            x_start = max(0, image_w - crop_w)
        y_start = max((image_h - crop_h) // 2, 0)
        y_end = y_start + crop_h
        if y_end > image_h:
            y_end = image_h
            y_start = max(0, image_h - crop_h)
        return y_start, y_end, x_start, x_end

    def scale_crop_bounds_to_target(
        self,
        bounds: tuple[int, int, int, int] | None,
        ref_hw: tuple[int, int] | None,
        target_hw: tuple[int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """把 HR 上的 crop 框缩放到 LR/SR/误差图尺寸，保证裁的是同一物理区域。"""

        if bounds is None or ref_hw is None or target_hw is None:
            return None
        ref_h, ref_w = ref_hw
        target_h, target_w = target_hw
        if min(ref_h, ref_w, target_h, target_w) <= 0:
            return None
        y0, y1, x0, x1 = bounds
        ty0 = int(round(y0 * target_h / ref_h))
        ty1 = int(round(y1 * target_h / ref_h))
        tx0 = int(round(x0 * target_w / ref_w))
        tx1 = int(round(x1 * target_w / ref_w))
        ty0 = max(0, min(target_h - 1, ty0))
        tx0 = max(0, min(target_w - 1, tx0))
        ty1 = max(ty0 + 1, min(target_h, ty1))
        tx1 = max(tx0 + 1, min(target_w, tx1))
        return ty0, ty1, tx0, tx1

    def crop_array_by_bounds(
        self, array: np.ndarray | None, bounds: tuple[int, int, int, int] | None
    ) -> np.ndarray | None:
        """按 y_start/y_end/x_start/x_end 裁剪二维或 HWC 图像数组。"""

        if array is None or bounds is None:
            return None
        arr = ensure_2d_image(array)
        if arr.ndim < 2:
            return None
        y0, y1, x0, x1 = bounds
        return np.ascontiguousarray(arr[y0:y1, x0:x1, ...])

    def draw_tbl_particle_crop_box(
        self,
        ax: plt.Axes,
        array: np.ndarray | None,
        ref_hw: tuple[int, int] | None,
        ref_bounds: tuple[int, int, int, int] | None,
    ) -> None:
        """在 TBL full-frame 颗粒图/误差图上画 crop 红框；crop 图本身不再重复画框。"""

        if array is None or ref_hw is None or ref_bounds is None:
            return
        target_hw = self.image_hw(array)
        bounds = self.scale_crop_bounds_to_target(ref_bounds, ref_hw, target_hw)
        if bounds is None:
            return
        y0, y1, x0, x1 = bounds
        patch_cls = ensure_matplotlib().matplotlib.patches.Rectangle
        ax.add_patch(
            patch_cls(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=getattr(self.cfg, "TBL_PARTICLE_CROP_BOX_COLOR", "red"),
                linewidth=float(getattr(self.cfg, "TBL_PARTICLE_CROP_BOX_LINE_WIDTH", 1.3)),
                zorder=8,
            )
        )

    def first_available_particle_crop_reference(self, sample_dir: Path, time_name: str) -> np.ndarray | None:
        """TBL crop 框优先以 GT 图为参考；GT 缺失时回退到 SR/LR，尽量不让 crop 图整体缺失。"""

        for key in ("gt", "sr", "lr"):
            array = self.load_particle_array(sample_dir, time_name, key)
            if array is not None:
                return array
        return None

    def load_particle_crop_array(self, sample_dir: Path, time_name: str, key: str) -> np.ndarray | None:
        """读取 TBL 颗粒 crop：已有 crop npy 直接读，SR/GT/LR 则按全局 crop 框从 full-frame npy 裁取。"""

        crop_path = self.particle_crop_path(sample_dir, time_name, key)
        crop_array = load_npy(crop_path)
        if crop_array is not None:
            return crop_array

        full_array = self.load_particle_array(sample_dir, time_name, key)
        if full_array is None:
            return None
        ref_array = self.first_available_particle_crop_reference(sample_dir, time_name)
        ref_hw = self.image_hw(ref_array)
        target_hw = self.image_hw(full_array)
        ref_bounds = self.resolve_tbl_particle_crop_bounds(ref_hw)
        bounds = self.scale_crop_bounds_to_target(ref_bounds, ref_hw, target_hw)
        return self.crop_array_by_bounds(full_array, bounds)

    def load_particle_array_mode(self, sample_dir: Path, time_name: str, key: str, crop: bool = False) -> np.ndarray | None:
        """按 full-frame/crop 模式读取颗粒数组，TBL crop 图复用同一套绘图函数。"""

        if crop:
            return self.load_particle_crop_array(sample_dir, time_name, key)
        return self.load_particle_array(sample_dir, time_name, key)

    def load_vorticity_maps(self, sample_dir: Path) -> dict[str, np.ndarray]:
        maps: dict[str, np.ndarray] = {}
        for key, file_name in self.cfg.VORTICITY_ARRAY_FILE_NAMES.items():
            array = load_npy(sample_dir / file_name)
            if array is not None:
                maps[key] = np.squeeze(array).astype(np.float32)
        if "error" not in maps and "pred" in maps and "gt" in maps:
            common_h = min(maps["pred"].shape[0], maps["gt"].shape[0])
            common_w = min(maps["pred"].shape[1], maps["gt"].shape[1])
            maps["error"] = maps["pred"][:common_h, :common_w] - maps["gt"][:common_h, :common_w]
        return maps

    def load_vorticity_quiver_flow(self, sample_dir: Path) -> np.ndarray | None:
        """读取涡度位移图上叠加的速度/位移箭头，优先使用预测光流 fake_flo.npy。"""

        pred, _ = self.load_flow_pair(sample_dir)
        if pred is not None:
            return pred
        for file_name in ("delta_uvw.npy", "delta_velocity_uv.npy"):
            array = load_npy(sample_dir / file_name)
            if array is not None:
                flow = flow_to_hw2(array)
                if flow is not None:
                    return flow
        return None

    # =========================
    # (6)(7)(8)(10) 组合图
    # =========================
    def plot_composite_bundle(self, group: GroupContext) -> None:
        """统一调度颗粒、光流、涡度和颗粒统计组合图。"""

        self.plot_particle_sr_error_composites(group)
        if normalize_name(group.category_name) == "experiment":
            self.plot_experiment_particle_zoom_composites(group)
        self.plot_flow_value_error_composites(group)
        self.plot_vorticity_composites(group)
        self.plot_particle_stats_composites(group)

    def plot_particle_stats_metric_only(self, group: GroupContext) -> None:
        """只生成颗粒统计条形统计图，不生成颗粒图/阈值图等其它组合图。"""

        self.plot_particle_stats_composites(group, metrics_only=True)

    def limited_bundles(self, group: GroupContext, kind: str) -> list[SampleBundle]:
        bundles = self.bundle_samples(group, kind)
        limit = self.cfg.MAX_SAMPLE_COMPOSITES_PER_CATEGORY
        if limit is not None:
            return bundles[: int(limit)]
        return bundles

    def panel_text(self, ax: plt.Axes, text: str, fontsize: float | None = None, loc: str = "upper_left") -> None:
        """用轴内文本作为面板 label，避免使用 Matplotlib title。"""

        if not text:
            return
        # 颗粒阈值化图的 label 往往是 “实验名 + binary”，比普通面板标题更长；
        # 这里允许 draw_map 传入较小字号，防止标题文字超出图像本身。
        label_size = self.cfg.PANEL_LABEL_SIZE if fontsize is None else fontsize
        # 个别 TBL full-frame 图左上角有 crop 红框，label 放左上会遮挡；
        # 因此支持把 label 放到右上，其它图默认仍使用左上角，保持旧版论文图样式。
        loc_key = normalize_name(loc)
        if loc_key in ("upper_right", "right_top", "top_right"):
            x, ha = 0.98, "right"
        else:
            x, ha = 0.02, "left"
        ax.text(
            x,
            0.98,
            text,
            transform=ax.transAxes,
            ha=ha,
            va="top",
            fontsize=label_size,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
        )

    def draw_map(
        self,
        ax: plt.Axes,
        array: np.ndarray | None,
        cmap: str,
        vmin: float | None,
        vmax: float | None,
        label: str,
        fill_panel: bool = False,
        label_fontsize: float | None = None,
        label_loc: str = "upper_left",
    ):
        if array is None:
            ax.axis("off")
            return None
        arr = ensure_2d_image(array)
        aspect = "auto" if fill_panel else None
        if arr.ndim == 2:
            handle = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
        else:
            handle = ax.imshow(arr, aspect=aspect)
        # label_fontsize 允许特定组合图显式指定更小字号；
        # 未指定时，仍对 binary 阈值图自动使用颗粒阈值专用字号。
        if label_fontsize is None and "binary" in str(label).lower():
            label_fontsize = float(getattr(self.cfg, "PARTICLE_BINARY_PANEL_LABEL_SIZE", self.cfg.PANEL_LABEL_SIZE))
        self.panel_text(ax, label, fontsize=label_fontsize, loc=label_loc)
        ax.axis("off")
        return handle

    def draw_quiver_overlay(self, ax: plt.Axes, flow: np.ndarray | None, image_array: np.ndarray | None) -> None:
        """在涡度底图上叠加位移/速度场箭头；箭头步长和样式均由 global_class 控制。"""

        if flow is None or image_array is None:
            return
        flow_hw2 = flow_to_hw2(flow)
        image_shape = self.image_hw(image_array)
        if flow_hw2 is None or image_shape is None:
            return
        img_h, img_w = image_shape
        common_h = min(img_h, flow_hw2.shape[0])
        common_w = min(img_w, flow_hw2.shape[1])
        stride = max(1, int(self.cfg.VORTICITY_QUIVER_STRIDE))
        y_idx = np.arange(stride // 2, common_h, stride)
        x_idx = np.arange(stride // 2, common_w, stride)
        if y_idx.size == 0 or x_idx.size == 0:
            return
        grid_x, grid_y = np.meshgrid(x_idx, y_idx)
        u = flow_hw2[np.ix_(y_idx, x_idx, [0])][..., 0]
        v = flow_hw2[np.ix_(y_idx, x_idx, [1])][..., 0]
        kwargs = {
            "color": self.cfg.VORTICITY_QUIVER_COLOR,
            "alpha": self.cfg.VORTICITY_QUIVER_ALPHA,
            "width": self.cfg.VORTICITY_QUIVER_WIDTH,
            "headwidth": self.cfg.VORTICITY_QUIVER_HEADWIDTH,
            "headlength": self.cfg.VORTICITY_QUIVER_HEADLENGTH,
            "angles": "xy",
            "scale_units": "xy",
            "zorder": 5,
        }
        if self.cfg.VORTICITY_QUIVER_SCALE is not None:
            kwargs["scale"] = self.cfg.VORTICITY_QUIVER_SCALE
        ax.quiver(grid_x, grid_y, u, v, **kwargs)

    def image_hw(self, array: np.ndarray | None) -> tuple[int, int] | None:
        """返回图像的高宽；用于让 LR 按原始像素尺寸放在 HR/SR 参考画布中。"""

        if array is None:
            return None
        arr = ensure_2d_image(array)
        if arr.ndim < 2:
            return None
        return int(arr.shape[0]), int(arr.shape[1])

    def reference_hw(self, arrays: Iterable[np.ndarray | None]) -> tuple[int, int] | None:
        """取一组 HR/SR 图里的最大高宽作为参考画布，LR 不再被拉伸到这个尺寸。"""

        shapes = [shape for shape in (self.image_hw(array) for array in arrays) if shape is not None]
        if not shapes:
            return None
        return max(h for h, _ in shapes), max(w for _, w in shapes)

    def draw_map_original_size(
        self,
        ax: plt.Axes,
        array: np.ndarray | None,
        cmap: str,
        vmin: float | None,
        vmax: float | None,
        label: str,
        reference_shape: tuple[int, int] | None,
    ):
        """
        按原始像素尺寸绘制 LR 图。
        普通 imshow 会把 LR 自动拉伸到和其它面板一样大；这里把 LR 放到 HR/SR 的参考画布上方居中位置，
        因而 LR 保持原大小，同时顶部与旁边的 GT/SR 图像对齐，左右方向按参考画布居中。
        """

        if array is None:
            ax.axis("off")
            return None
        arr = ensure_2d_image(array)
        own_shape = self.image_hw(arr)
        if own_shape is None:
            ax.axis("off")
            return None
        ref_h, ref_w = reference_shape or own_shape
        own_h, own_w = own_shape
        # LR 的纵向起点固定为 0，实现“上对齐”；横向按参考画布居中，实现用户要求的“上居中对齐”。
        x0 = max(0.0, (float(ref_w) - float(own_w)) / 2.0)
        x1 = x0 + float(own_w)
        if arr.ndim == 2:
            handle = ax.imshow(
                arr,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(x0, x1, own_h, 0),
                interpolation="nearest",
            )
        else:
            handle = ax.imshow(
                arr,
                extent=(x0, x1, own_h, 0),
                interpolation="nearest",
            )
        ax.set_xlim(0, ref_w)
        ax.set_ylim(ref_h, 0)
        ax.set_aspect("equal", adjustable="box")
        self.panel_text(ax, label)
        ax.axis("off")
        return handle

    def experiment_zoom_bounds(
        self,
        region: tuple[float, float, float, float],
        height: int,
        width: int,
    ) -> tuple[int, int, int, int]:
        """把 experiment 局部放大相对区域换算成 y0/y1/x0/x1。"""

        x_ratio, y_ratio, w_ratio, h_ratio = region
        crop_w = max(8, int(round(width * float(w_ratio))))
        crop_h = max(8, int(round(height * float(h_ratio))))
        cx = int(round(width * float(x_ratio)))
        cy = int(round(height * float(y_ratio)))
        x0 = max(0, min(width - crop_w, cx - crop_w // 2))
        y0 = max(0, min(height - crop_h, cy - crop_h // 2))
        return y0, y0 + crop_h, x0, x0 + crop_w

    def scale_zoom_bounds(
        self,
        bounds: tuple[int, int, int, int],
        source_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """把 HR/SR 坐标框缩放到 LR 坐标。"""

        y0, y1, x0, x1 = bounds
        src_h, src_w = source_hw
        dst_h, dst_w = target_hw
        return (
            max(0, min(dst_h - 1, int(round(y0 * dst_h / src_h)))),
            max(1, min(dst_h, int(round(y1 * dst_h / src_h)))),
            max(0, min(dst_w - 1, int(round(x0 * dst_w / src_w)))),
            max(1, min(dst_w, int(round(x1 * dst_w / src_w)))),
        )

    def draw_zoom_overview(self, ax: plt.Axes, array: np.ndarray | None, bounds_list, label: str):
        """绘制带红框的整图 overview。"""

        handle = self.draw_map(ax, array, self.cfg.IMAGE_CMAP, 0.0, 1.0, label)
        if array is not None:
            import matplotlib.patches as patches

            for idx, (y0, y1, x0, x1) in enumerate(bounds_list, start=1):
                ax.add_patch(
                    patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.2)
                )
                ax.text(
                    x0,
                    y0,
                    str(idx),
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": "red", "edgecolor": "none", "pad": 1},
                )
        return handle

    def row_limit(
        self,
        arrays: Iterable[np.ndarray | None],
        configured_limit,
        center_zero: bool = False,
    ) -> tuple[float | None, float | None]:
        valid = [ensure_2d_image(a) for a in arrays if a is not None and ensure_2d_image(a).ndim == 2]
        if not valid:
            return None, None
        return self.resolve_color_limit(valid, configured_limit, center_zero=center_zero)

    def flow_value_component_fallback_limit(self, category_name: str, component: str) -> tuple[float, float] | None:
        """读取光流值图的兜底色条范围；用于只有 RGB 裁剪面板、没有原始二维数值场的 test_all。"""

        limits_by_category = getattr(self.cfg, "FLOW_VALUE_COMPONENT_FALLBACK_LIMITS", {})
        category_key = normalize_name(category_name)
        component_key = normalize_name(component)
        category_limits = limits_by_category.get(category_key) or limits_by_category.get("default") or {}
        limit = category_limits.get(component_key)
        if isinstance(limit, (tuple, list)) and len(limit) == 2:
            return float(limit[0]), float(limit[1])
        return None

    def colorbar_mappable(self, handle, cmap: str, vmin: float, vmax: float):
        """为色条准备 mappable；RGB 面板不能直接生成物理色条时，使用同色图和给定范围构造色条。"""

        try:
            array = handle.get_array()
            if array is not None and np.asarray(array).ndim == 2:
                return handle
        except Exception:
            pass
        mpl = ensure_matplotlib().matplotlib
        return mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    def composite_wrap_method_count(self) -> int | None:
        """读取当前对比组组合图每块显示的实验数量；None 表示不换行。"""

        mapping = getattr(self.cfg, "COMPARISON_GROUP_COMPOSITE_WRAP_METHOD_COUNT", {})
        value = mapping.get(self.active_comparison_name)
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def chunk_items(self, items: list, chunk_size: int | None) -> list[list]:
        """把实验列表按 chunk_size 分块；chunk_size=None 时保持一个整体。"""

        if not chunk_size or chunk_size >= len(items):
            return [items]
        return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]

    def tbl_profile_valid_height(self, sample_dir: Path) -> int | None:
        """读取 TBL profile_analysis 中保存的有效 y 高度，用于裁掉误差图底部无效壁面区。"""

        profile_root_name = getattr(self.cfg, "TBL_PROFILE_DIR_NAME", "profile_analysis")
        # flow 样本可能位于 category/sample_0000，也可能位于 category/flow/sample_0000；
        # 因此同时检查 parent 和 parent.parent，找到与当前 sample 同名的 profile_analysis 目录。
        candidate_dirs = [
            sample_dir.parent / profile_root_name / sample_dir.name,
            sample_dir.parent.parent / profile_root_name / sample_dir.name,
        ]
        profile_dir = next((path for path in candidate_dirs if path.exists()), candidate_dirs[0])
        limit = load_npy(profile_dir / "profile_y_limit.npy")
        if limit is not None:
            try:
                value = int(np.asarray(limit).reshape(-1)[0])
                return value if value > 0 else None
            except Exception:
                return None
        y_positions = load_npy(profile_dir / self.tbl_profile_file_name("y", "u"))
        if y_positions is not None:
            size = int(np.asarray(y_positions).size)
            return size if size > 0 else None
        return None

    def trim_tbl_flow_error_bottom(self, array: np.ndarray | None, sample_dir: Path | None = None) -> np.ndarray | None:
        """裁掉 TBL 光流误差图底部由无效区域形成的连续近似常值矩形带。"""

        if array is None or not getattr(self.cfg, "TBL_FLOW_ERROR_TRIM_BOTTOM_ENABLED", True):
            return array
        arr = ensure_2d_image(array)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return array
        # TBL 生成 profile_analysis 时已经把有效边界层高度保存为 profile_y_limit.npy。
        # flow_u/v/s_value_error_composite 的底部矩形正是这个有效区域之外的壁面/填充值，
        # 因此优先按该高度裁剪；只在旧目录没有 profile 文件时才使用下面的自动检测。
        if sample_dir is not None:
            valid_height = self.tbl_profile_valid_height(sample_dir)
            if valid_height is not None and 0 < valid_height < arr.shape[0]:
                return arr[:valid_height, :]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return array
        height = arr.shape[0]
        max_rows = max(1, int(height * float(getattr(self.cfg, "TBL_FLOW_ERROR_TRIM_MAX_FRACTION", 0.28))))
        min_rows = max(1, int(getattr(self.cfg, "TBL_FLOW_ERROR_TRIM_MIN_ROWS", 4)))
        global_std = float(np.nanstd(finite))
        abs_scale = float(np.nanpercentile(np.abs(finite), 90))
        std_threshold = max(global_std * float(getattr(self.cfg, "TBL_FLOW_ERROR_TRIM_STD_RATIO", 0.18)), 1e-8)
        mean_threshold = max(abs_scale * float(getattr(self.cfg, "TBL_FLOW_ERROR_TRIM_MEAN_RATIO", 0.20)), 1e-8)
        trim_rows = 0
        for offset in range(1, max_rows + 1):
            row = arr[-offset, :]
            row_finite = row[np.isfinite(row)]
            if row_finite.size == 0:
                trim_rows += 1
                continue
            # 底部无效带通常是一整行几乎同色的红/蓝误差，表现为低方差且均值绝对值较大。
            # 只有从最底部连续满足条件时才裁掉，避免误删正常流场中的局部结构。
            row_std = float(np.nanstd(row_finite))
            row_mean_abs = abs(float(np.nanmean(row_finite)))
            if row_std <= std_threshold and row_mean_abs >= mean_threshold:
                trim_rows += 1
                continue
            break
        if trim_rows >= min_rows and trim_rows < height:
            return arr[: height - trim_rows, :]
        return array

    def plot_particle_sr_error_composites(self, group: GroupContext) -> None:
        """图六：TBL full-frame 按最新要求恢复横向排版，并额外输出 crop 横向排版图。"""

        is_tbl = normalize_name(group.category_name) == "tbl"
        if is_tbl and getattr(self.cfg, "TBL_FULL_FRAME_VERTICAL_LAYOUT", False):
            self.plot_particle_sr_error_composites_vertical(group)
            if getattr(self.cfg, "TBL_PARTICLE_CROP_ENABLED", True):
                self.plot_particle_sr_error_composites_horizontal(group, crop=True)
            return
        self.plot_particle_sr_error_composites_horizontal(group, crop=False)
        if is_tbl and getattr(self.cfg, "TBL_PARTICLE_CROP_ENABLED", True):
            self.plot_particle_sr_error_composites_horizontal(group, crop=True)

    def plot_experiment_particle_zoom_composites(self, group: GroupContext) -> None:
        """experiment 专用：跨实验局部放大颗粒对比图，并在整图上标红框。"""

        regions = tuple(getattr(self.cfg, "EXPERIMENT_PARTICLE_ZOOM_REGIONS", ()))
        if not regions:
            return
        for bundle in self.limited_bundles(group, "particle"):
            exp_keys = [key for key in group.experiment_keys if key in bundle.sample_dirs]
            if not exp_keys:
                continue
            for time_name in ("previous", "next"):
                gt = self.first_available_particle(bundle, time_name, "gt", crop=False)
                sr_maps = {
                    exp_key: self.load_particle_array_mode(sample_dir, time_name, "sr", crop=False)
                    for exp_key, sample_dir in bundle.sample_dirs.items()
                }
                if gt is None or not any(value is not None for value in sr_maps.values()):
                    continue
                gt_arr = ensure_2d_image(gt)
                gt_hw = self.image_hw(gt_arr)
                if gt_hw is None:
                    continue
                gt_h, gt_w = gt_hw
                bounds_list = [self.experiment_zoom_bounds(region, gt_h, gt_w) for region in regions]

                # experiment 的 zoom composite 只展示 GT 和各方法 SR。
                # LR 尺寸与 SR/GT 不同，SR-HR 误差又基于插值伪 GT；去掉它们后，
                # 每个局部块可以画得更大，红框区域里的颗粒细节更容易直接比较。
                n_cols = 1 + len(exp_keys)
                n_rows = 1 + len(bounds_list)
                fig = plt.figure(figsize=(2.65 * n_cols + 0.8, 2.45 * n_rows + 0.4))
                gs = fig.add_gridspec(n_rows, n_cols, hspace=0.08, wspace=0.035)

                self.draw_zoom_overview(fig.add_subplot(gs[0, 0]), gt_arr, bounds_list, self.cfg.GT_PANEL_LABEL)
                for col_idx, exp_key in enumerate(exp_keys, start=1):
                    self.draw_zoom_overview(
                        fig.add_subplot(gs[0, col_idx]),
                        sr_maps.get(exp_key),
                        bounds_list,
                        self.experiment_label(exp_key),
                    )

                for row_idx, bounds in enumerate(bounds_list, start=1):
                    y0, y1, x0, x1 = bounds
                    self.draw_map(
                        fig.add_subplot(gs[row_idx, 0]),
                        gt_arr[y0:y1, x0:x1],
                        self.cfg.IMAGE_CMAP,
                        0.0,
                        1.0,
                        f"R{row_idx} GT",
                    )
                    for col_idx, exp_key in enumerate(exp_keys, start=1):
                        sr = sr_maps.get(exp_key)
                        sr_crop = self.crop_array_by_bounds(sr, bounds) if sr is not None else None
                        self.draw_map(
                            fig.add_subplot(gs[row_idx, col_idx]),
                            sr_crop,
                            self.cfg.IMAGE_CMAP,
                            0.0,
                            1.0,
                            self.experiment_label(exp_key),
                        )

                out_dir = self.output_dir(
                    self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                    group.class_name,
                    group.split_name,
                    group.category_name,
                    bundle.sample_name,
                )
                self.save_figure(fig, out_dir / f"particle_zoom_composite_{time_name}")

    def plot_particle_sr_error_composites_horizontal(self, group: GroupContext, crop: bool = False) -> None:
        """previous/next 的 LR、GT、八个 SR 与对应误差图；crop=True 时使用 TBL crop 数据但保持横向布局。"""

        for bundle in self.limited_bundles(group, "particle"):
            rows = []
            for time_name in ("previous", "next"):
                lr = self.first_available_particle(bundle, time_name, "lr", crop=crop)
                gt = self.first_available_particle(bundle, time_name, "gt", crop=crop)
                sr_maps = {
                    exp_key: self.load_particle_array_mode(sample_dir, time_name, "sr", crop=crop)
                    for exp_key, sample_dir in bundle.sample_dirs.items()
                }
                err_maps = {
                    exp_key: self.load_particle_array_mode(sample_dir, time_name, "error", crop=crop)
                    for exp_key, sample_dir in bundle.sample_dirs.items()
                }
                rows.append((time_name, "image", [lr, gt], sr_maps))
                rows.append((time_name, "error", [None, None], err_maps))

            exp_keys = [key for key in group.experiment_keys if key in bundle.sample_dirs]
            exp_chunks = self.chunk_items(exp_keys, self.composite_wrap_method_count())
            fixed_count = 2
            method_cols = max((len(chunk) for chunk in exp_chunks), default=0)
            n_cols = fixed_count + method_cols
            row_specs = []
            for time_name, row_kind, fixed_maps, exp_maps in rows:
                for chunk_idx, chunk in enumerate(exp_chunks):
                    row_specs.append((time_name, row_kind, fixed_maps, exp_maps, chunk_idx, chunk))
            fig = plt.figure(figsize=(2.0 * n_cols + 0.5, max(4.2, 2.05 * len(row_specs))))
            gs = fig.add_gridspec(
                len(row_specs),
                n_cols + 1,
                width_ratios=[1] * n_cols + [0.06],
                hspace=0.08,
                wspace=0.04,
            )
            for row_idx, (time_name, row_kind, fixed_maps, exp_maps, chunk_idx, chunk) in enumerate(row_specs):
                show_fixed = chunk_idx == 0
                # 颗粒图第一块保留 LR/GT；eight_experiments 的第二块只空出最左侧参考列，
                # 让后 4 个实验从第二列开始，避免继续对齐到 GT/LR 两个固定列之后导致画面右移。
                current_fixed_maps = fixed_maps if show_fixed else [None]
                row_arrays = current_fixed_maps + [exp_maps.get(exp_key) for exp_key in chunk]
                row_arrays += [None] * (n_cols - len(row_arrays))
                # 颗粒 SR 对比图的 LR 面板使用原始低分辨率尺寸；
                # 参考画布取 GT 和各个 SR 的最大尺寸，确保 LR 顶部与旁边图像上边缘对齐。
                lr_reference_shape = None
                if row_kind == "image":
                    lr_reference_shape = self.reference_hw([fixed_maps[1]] + [exp_maps.get(exp_key) for exp_key in exp_keys])
                if row_kind == "image":
                    all_row_arrays = fixed_maps + [exp_maps.get(exp_key) for exp_key in exp_keys]
                    vmin, vmax = self.row_limit(all_row_arrays, self.cfg.PARTICLE_VALUE_COLORBAR_LIMIT)
                    cmap = self.cfg.IMAGE_CMAP
                    colorbar_label = self.cfg.PARTICLE_VALUE_COLORBAR_LABEL
                    labels = ([self.cfg.LR_PANEL_LABEL, self.cfg.GT_PANEL_LABEL] if show_fixed else [""]) + [
                        self.experiment_label(exp_key) for exp_key in chunk
                    ]
                else:
                    # 组合图中的颗粒误差行也使用 0 居中的白色发散色条。
                    vmin, vmax = self.row_limit(
                        self.error_colorbar_reference_arrays(exp_maps),
                        self.cfg.PARTICLE_ERROR_COLORBAR_LIMIT,
                        center_zero=True,
                    )
                    cmap = self.cfg.ERROR_CMAP
                    colorbar_label = self.cfg.PARTICLE_ERROR_COLORBAR_LABEL
                    labels = ([self.cfg.BLANK_PANEL_LABEL, self.cfg.BLANK_PANEL_LABEL] if show_fixed else [""]) + [
                        self.experiment_label(exp_key) for exp_key in chunk
                    ]
                labels += [""] * (n_cols - len(labels))
                image_handle = None
                for col_idx, array in enumerate(row_arrays):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    if row_kind == "image" and col_idx == 0:
                        handle = self.draw_map_original_size(
                            ax,
                            array,
                            cmap,
                            vmin,
                            vmax,
                            labels[col_idx],
                            lr_reference_shape,
                        )
                    else:
                        fill_panel = (
                            normalize_name(group.category_name) == "tbl"
                            and not crop
                            and row_kind == "error"
                            and bool(getattr(self.cfg, "TBL_FULL_FRAME_ERROR_FILL_PANEL", True))
                        )
                        handle = self.draw_map(ax, array, cmap, vmin, vmax, labels[col_idx], fill_panel=fill_panel)
                    image_handle = handle or image_handle
                    if col_idx == 0 and show_fixed:
                        ax.text(
                            -0.08,
                            0.5,
                            self.cfg.PREVIOUS_ROW_LABEL if time_name == "previous" else self.cfg.NEXT_ROW_LABEL,
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            rotation=90,
                            fontsize=self.cfg.PANEL_LABEL_SIZE,
                        )
                cax = fig.add_subplot(gs[row_idx, n_cols])
                if image_handle is not None and vmin is not None and vmax is not None:
                    cb = fig.colorbar(image_handle, cax=cax)
                    cb.set_label(colorbar_label, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax.axis("off")

            out_dir = self.output_dir(
                self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                group.class_name,
                group.split_name,
                group.category_name,
                bundle.sample_name,
            )
            suffix = getattr(self.cfg, "TBL_PARTICLE_CROP_OUTPUT_SUFFIX", "_crop") if crop else ""
            self.save_figure(fig, out_dir / f"particle_sr_error_composite{suffix}")

    def plot_particle_sr_error_composites_vertical(self, group: GroupContext) -> None:
        """TBL full-frame 颗粒图专用竖排版：每行一个实验/参考图，左列图像，右列误差。"""

        for bundle in self.limited_bundles(group, "particle"):
            blocks = []
            value_arrays: list[np.ndarray | None] = []
            error_maps_for_limit: dict[str, np.ndarray] = {}
            exp_keys = [key for key in self.legend_order_keys() if key in bundle.sample_dirs]
            if not exp_keys:
                continue

            for time_name in ("previous", "next"):
                lr = self.first_available_particle(bundle, time_name, "lr")
                gt = self.first_available_particle(bundle, time_name, "gt")
                sr_maps = {
                    exp_key: self.load_particle_array_mode(sample_dir, time_name, "sr")
                    for exp_key, sample_dir in bundle.sample_dirs.items()
                }
                err_maps = {
                    exp_key: self.load_particle_array_mode(sample_dir, time_name, "error")
                    for exp_key, sample_dir in bundle.sample_dirs.items()
                }
                value_arrays.extend([lr, gt])
                value_arrays.extend(sr_maps.get(exp_key) for exp_key in exp_keys)
                for exp_key, error in err_maps.items():
                    if error is not None:
                        error_maps_for_limit[f"{time_name}_{exp_key}"] = error
                blocks.append((time_name, lr, gt, sr_maps, err_maps))

            image_vmin, image_vmax = self.row_limit(value_arrays, self.cfg.PARTICLE_VALUE_COLORBAR_LIMIT)
            err_refs = [
                error_maps_for_limit[key]
                for key in error_maps_for_limit
                if any(key.endswith(f"_{ref}") for ref in getattr(self.cfg, "ERROR_COLORBAR_REFERENCE_EXPERIMENT_KEYS", ()))
            ]
            if not err_refs:
                err_refs = [value for value in error_maps_for_limit.values() if value is not None]
            error_vmin, error_vmax = self.row_limit(err_refs, self.cfg.PARTICLE_ERROR_COLORBAR_LIMIT, center_zero=True)

            row_items = []
            height_ratios = []
            for block_idx, (time_name, lr, gt, sr_maps, err_maps) in enumerate(blocks):
                if block_idx:
                    row_items.append(("gap", "", None, None))
                    height_ratios.append(0.45)
                row_label = self.cfg.PREVIOUS_ROW_LABEL if time_name == "previous" else self.cfg.NEXT_ROW_LABEL
                row_items.append((row_label, self.cfg.LR_PANEL_LABEL, lr, None))
                row_items.append(("", self.cfg.GT_PANEL_LABEL, gt, None))
                height_ratios.extend([1.0, 1.0])
                for exp_key in exp_keys:
                    row_items.append(("", self.experiment_label(exp_key), sr_maps.get(exp_key), err_maps.get(exp_key)))
                    height_ratios.append(1.0)

            fig_width = float(getattr(self.cfg, "TBL_FULL_FRAME_FIG_WIDTH", 7.2))
            row_height = float(getattr(self.cfg, "TBL_FULL_FRAME_ROW_HEIGHT", 1.65))
            fig_height = max(4.8, row_height * sum(height_ratios))
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = fig.add_gridspec(
                len(row_items),
                4,
                width_ratios=[1, 1, 0.045, 0.045],
                height_ratios=height_ratios,
                wspace=float(getattr(self.cfg, "TBL_FULL_FRAME_WSPACE", 0.08)),
                hspace=float(getattr(self.cfg, "TBL_FULL_FRAME_HSPACE", 0.10)),
            )

            for row_idx, (row_group_label, panel_label, image_array, error_array) in enumerate(row_items):
                if row_group_label == "gap":
                    # 间隔行四列全部关闭；TBL full-frame 颗粒图不再使用跨全图色条，
                    # 而是在每一行右侧单独放图像色条和误差色条，避免一条色条占满所有行。
                    for col_idx in range(4):
                        ax = fig.add_subplot(gs[row_idx, col_idx])
                        ax.axis("off")
                    continue
                ax_image = fig.add_subplot(gs[row_idx, 0])
                row_image_handle = self.draw_map(
                    ax_image,
                    image_array,
                    self.cfg.IMAGE_CMAP,
                    image_vmin,
                    image_vmax,
                    panel_label,
                )
                if row_group_label:
                    ax_image.text(
                        -0.08,
                        0.5,
                        row_group_label,
                        transform=ax_image.transAxes,
                        ha="right",
                        va="center",
                        rotation=90,
                        fontsize=self.cfg.PANEL_LABEL_SIZE,
                    )
                ax_error = fig.add_subplot(gs[row_idx, 1])
                row_error_handle = self.draw_map(
                    ax_error,
                    error_array,
                    self.cfg.ERROR_CMAP,
                    error_vmin,
                    error_vmax,
                    panel_label if error_array is not None else "",
                )

                # 每一行都使用独立色条：第三列对应颗粒/SR 图像，第四列对应误差图。
                # LR/GT 行没有误差图时只关闭误差色条轴；实验行会同时显示图像色条和误差色条。
                cax_image = fig.add_subplot(gs[row_idx, 2])
                if row_image_handle is not None and image_vmin is not None and image_vmax is not None:
                    cb = fig.colorbar(row_image_handle, cax=cax_image)
                    cb.set_label(self.cfg.PARTICLE_VALUE_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax_image.axis("off")
                cax_error = fig.add_subplot(gs[row_idx, 3])
                if row_error_handle is not None and error_vmin is not None and error_vmax is not None:
                    cb = fig.colorbar(row_error_handle, cax=cax_error)
                    cb.set_label(self.cfg.PARTICLE_ERROR_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax_error.axis("off")

            out_dir = self.output_dir(
                self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                group.class_name,
                group.split_name,
                group.category_name,
                bundle.sample_name,
            )
            self.save_figure(fig, out_dir / "particle_sr_error_composite")

    def first_available_particle(
        self, bundle: SampleBundle, time_name: str, key: str, crop: bool = False
    ) -> np.ndarray | None:
        for sample_dir in bundle.sample_dirs.values():
            array = self.load_particle_array_mode(sample_dir, time_name, key, crop=crop)
            if array is not None:
                return array
        return None

    def plot_flow_value_error_composites(self, group: GroupContext) -> None:
        """图七：u/v/s 三个方向的 GT、预测光流与误差图组合。"""

        for bundle in self.limited_bundles(group, "flow"):
            is_tbl = normalize_name(group.category_name) == "tbl"
            for component in ("u", "v", "s"):
                gt_map = None
                pred_maps: dict[str, np.ndarray] = {}
                err_maps: dict[str, np.ndarray] = {}
                for exp_key, sample_dir in bundle.sample_dirs.items():
                    value_maps = self.load_flow_value_maps(sample_dir)
                    error_maps = self.load_flow_error_maps(sample_dir)
                    gt_map = gt_map if gt_map is not None else value_maps.get(f"gt_{component}")
                    if f"pred_{component}" in value_maps:
                        pred_maps[exp_key] = value_maps[f"pred_{component}"]
                    if component in error_maps:
                        err_map = error_maps[component]
                        if is_tbl:
                            err_map = self.trim_tbl_flow_error_bottom(err_map, sample_dir)
                        err_maps[exp_key] = err_map
                if not pred_maps and not err_maps:
                    continue
                self.plot_two_row_method_composite(
                    fixed_top=[gt_map],
                    method_top=pred_maps,
                    fixed_bottom=[None],
                    method_bottom=err_maps,
                    out_base=self.output_dir(
                        self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                        group.class_name,
                        group.split_name,
                        group.category_name,
                        bundle.sample_name,
                    )
                    / f"flow_{component}_value_error_composite",
                    top_cmap=self.cfg.IMAGE_CMAP,
                    bottom_cmap=self.cfg.ERROR_CMAP,
                    top_limit=self.cfg.FLOW_VALUE_COLORBAR_LIMIT,
                    bottom_limit=self.cfg.FLOW_ERROR_COLORBAR_LIMIT,
                    top_fallback_limit=self.flow_value_component_fallback_limit(group.category_name, component),
                    top_colorbar_label=self.cfg.FLOW_VALUE_COLORBAR_LABEL,
                    bottom_colorbar_label=self.cfg.FLOW_ERROR_COLORBAR_LABEL,
                    fixed_top_labels=[self.cfg.FLOW_GT_PANEL_LABEL],
                    fixed_bottom_labels=[self.cfg.BLANK_PANEL_LABEL],
                    method_labels={k: self.experiment_label(k) for k in pred_maps.keys() | err_maps.keys()},
                    bottom_center_zero=True,
                    bottom_reference_arrays=self.error_colorbar_reference_arrays(err_maps),
                    bottom_fill_panel=is_tbl,
                )

    def plot_vorticity_composites(self, group: GroupContext) -> None:
        """图八：GT 原涡度图、八个实验的涡度位移图与涡度误差图。"""

        for bundle in self.limited_bundles(group, "vorticity"):
            gt_map = None
            gt_quiver = None
            pred_maps: dict[str, np.ndarray] = {}
            err_maps: dict[str, np.ndarray] = {}
            quiver_maps: dict[str, np.ndarray] = {}
            for exp_key, sample_dir in bundle.sample_dirs.items():
                maps = self.load_vorticity_maps(sample_dir)
                gt_map = gt_map if gt_map is not None else maps.get("gt")
                if "pred" in maps:
                    pred_maps[exp_key] = maps["pred"]
                if "error" in maps:
                    err_maps[exp_key] = maps["error"]
                pred_flow, gt_flow = self.load_flow_pair(sample_dir)
                if gt_quiver is None and gt_flow is not None:
                    gt_quiver = gt_flow
                if pred_flow is not None:
                    quiver_maps[exp_key] = pred_flow
                else:
                    # 个别旧结果没有 fake_flo/hr_flo 时，仍回退到已有 delta 位移场，保证涡度箭头尽量可画。
                    flow = self.load_vorticity_quiver_flow(sample_dir)
                    if flow is not None:
                        quiver_maps[exp_key] = flow
            if not pred_maps and not err_maps:
                continue
            exp_keys = [key for key in self.legend_order_keys() if key in pred_maps or key in err_maps]
            if not exp_keys:
                continue

            # 八组对比横向排满会过挤，因此复用组合图分块策略：
            # 第一块显示 GT + 前 4 个实验，第二块 GT 位置留空，让后 4 个实验从第二列开始。
            # 色条按每一行单独生成，保证换行后每块图仍有对应的物理量说明。
            exp_chunks = self.chunk_items(exp_keys, self.composite_wrap_method_count())
            top_vmin, top_vmax = self.row_limit(
                [gt_map] + [pred_maps.get(exp_key) for exp_key in exp_keys],
                self.cfg.VORTICITY_VALUE_COLORBAR_LIMIT,
            )
            bottom_vmin, bottom_vmax = self.row_limit(
                [err_maps.get(exp_key) for exp_key in exp_keys],
                self.cfg.VORTICITY_ERROR_COLORBAR_LIMIT,
            )
            n_cols = 1 + max((len(chunk) for chunk in exp_chunks), default=0)
            n_rows = 2 * len(exp_chunks)
            fig = plt.figure(figsize=(2.05 * n_cols + 0.45, max(4.3, 2.15 * n_rows)))
            gs = fig.add_gridspec(
                n_rows,
                n_cols + 1,
                width_ratios=[1] * n_cols + [0.06],
                hspace=0.08,
                wspace=0.04,
            )

            for chunk_idx, chunk in enumerate(exp_chunks):
                top_row = chunk_idx * 2
                bottom_row = top_row + 1
                top_handle = None
                bottom_handle = None

                ax = fig.add_subplot(gs[top_row, 0])
                if chunk_idx == 0:
                    top_handle = self.draw_map(
                        ax,
                        gt_map,
                        self.cfg.IMAGE_CMAP,
                        top_vmin,
                        top_vmax,
                        self.cfg.VORTICITY_GT_PANEL_LABEL,
                    ) or top_handle
                    # GT 原图也叠加 GT 位移/速度场箭头，和各实验预测涡度位移图形成同一语义对比。
                    self.draw_quiver_overlay(ax, gt_quiver, gt_map)
                else:
                    ax.axis("off")

                ax = fig.add_subplot(gs[bottom_row, 0])
                ax.axis("off")

                for col_idx, exp_key in enumerate(chunk, start=1):
                    ax = fig.add_subplot(gs[top_row, col_idx])
                    top_handle = self.draw_map(
                        ax,
                        pred_maps.get(exp_key),
                        self.cfg.IMAGE_CMAP,
                        top_vmin,
                        top_vmax,
                        self.experiment_label(exp_key),
                    ) or top_handle
                    # 涡度位移图第一行叠加 fake_flo 位移/速度场箭头，和原始 vorticity_quiver 图保持语义一致。
                    self.draw_quiver_overlay(ax, quiver_maps.get(exp_key), pred_maps.get(exp_key))

                    ax = fig.add_subplot(gs[bottom_row, col_idx])
                    bottom_handle = self.draw_map(
                        ax,
                        err_maps.get(exp_key),
                        self.cfg.ERROR_CMAP,
                        bottom_vmin,
                        bottom_vmax,
                        self.experiment_label(exp_key),
                    ) or bottom_handle

                # 每块不足 4 个实验时，末尾空列关闭，避免留下带坐标轴的空面板。
                for empty_col in range(1 + len(chunk), n_cols):
                    fig.add_subplot(gs[top_row, empty_col]).axis("off")
                    fig.add_subplot(gs[bottom_row, empty_col]).axis("off")

                cax = fig.add_subplot(gs[top_row, n_cols])
                if top_handle is not None and top_vmin is not None and top_vmax is not None:
                    cb = fig.colorbar(top_handle, cax=cax)
                    cb.set_label(self.cfg.VORTICITY_VALUE_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax.axis("off")
                cax = fig.add_subplot(gs[bottom_row, n_cols])
                if bottom_handle is not None and bottom_vmin is not None and bottom_vmax is not None:
                    cb = fig.colorbar(bottom_handle, cax=cax)
                    cb.set_label(self.cfg.VORTICITY_ERROR_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax.axis("off")

            out_base = self.output_dir(
                self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                group.class_name,
                group.split_name,
                group.category_name,
                bundle.sample_name,
            ) / "vorticity_value_error_composite"
            self.save_figure(fig, out_base)

    def plot_two_row_method_composite(
        self,
        fixed_top: list[np.ndarray | None],
        method_top: dict[str, np.ndarray],
        fixed_bottom: list[np.ndarray | None],
        method_bottom: dict[str, np.ndarray],
        out_base: Path,
        top_cmap: str,
        bottom_cmap: str,
        top_limit,
        bottom_limit,
        top_fallback_limit,
        top_colorbar_label: str,
        bottom_colorbar_label: str,
        fixed_top_labels: list[str],
        fixed_bottom_labels: list[str],
        method_labels: dict[str, str],
        bottom_center_zero: bool = False,
        bottom_reference_arrays: Iterable[np.ndarray | None] | None = None,
        bottom_fill_panel: bool = False,
    ) -> None:
        """通用两行拼图：第一行数值图，第二行误差图，每行末尾一个统一色条。"""

        exp_keys = [key for key in self.legend_order_keys() if key in method_top or key in method_bottom]
        all_top_arrays = fixed_top + [method_top.get(exp_key) for exp_key in exp_keys]
        all_bottom_arrays = fixed_bottom + [method_bottom.get(exp_key) for exp_key in exp_keys]

        top_vmin, top_vmax = self.row_limit(all_top_arrays, top_limit)
        # test_all 的 TBL/TWCF 光流值图可能来自 uvs_compare.png 的 RGB 裁剪图；
        # RGB 图没有原始物理数值，row_limit 会拿不到范围。此时使用全局兜底范围，
        # 只补绘色条，不改变已经裁剪好的光流面板像素内容。
        if (top_vmin is None or top_vmax is None) and top_fallback_limit is not None:
            top_vmin, top_vmax = top_fallback_limit
        # 光流误差组合图需要 0 居中白色；其它误差图保持原配置。
        bottom_vmin, bottom_vmax = self.row_limit(
            list(bottom_reference_arrays) if bottom_reference_arrays is not None else all_bottom_arrays,
            bottom_limit,
            center_zero=bottom_center_zero,
        )

        chunks = self.chunk_items(exp_keys, self.composite_wrap_method_count())
        fixed_count = max(len(fixed_top), len(fixed_bottom))
        method_cols = max((len(chunk) for chunk in chunks), default=0)
        n_cols = fixed_count + method_cols
        if n_cols == 0:
            return
        row_count = 2 * len(chunks)
        fig = plt.figure(figsize=(2.05 * n_cols + 0.45, 2.15 * row_count))
        gs = fig.add_gridspec(
            row_count,
            n_cols + 1,
            width_ratios=[1] * n_cols + [0.06],
            hspace=0.12,
            wspace=0.04,
        )

        for chunk_idx, chunk in enumerate(chunks):
            # 第一块显示 GT 等固定列；后续块在固定列位置留空，因此后 4 个实验从第二列开始。
            show_fixed = chunk_idx == 0
            top_arrays = (fixed_top if show_fixed else [None] * len(fixed_top)) + [method_top.get(exp_key) for exp_key in chunk]
            bottom_arrays = (fixed_bottom if show_fixed else [None] * len(fixed_bottom)) + [method_bottom.get(exp_key) for exp_key in chunk]
            top_labels = (fixed_top_labels if show_fixed else [""] * len(fixed_top_labels)) + [
                method_labels.get(exp_key, self.experiment_label(exp_key)) for exp_key in chunk
            ]
            bottom_labels = (fixed_bottom_labels if show_fixed else [""] * len(fixed_bottom_labels)) + [
                method_labels.get(exp_key, self.experiment_label(exp_key)) for exp_key in chunk
            ]
            top_arrays += [None] * (n_cols - len(top_arrays))
            bottom_arrays += [None] * (n_cols - len(bottom_arrays))
            top_labels += [""] * (n_cols - len(top_labels))
            bottom_labels += [""] * (n_cols - len(bottom_labels))

            for local_row_idx, (arrays, labels, cmap, vmin, vmax, cb_label) in enumerate(
                (
                    (top_arrays, top_labels, top_cmap, top_vmin, top_vmax, top_colorbar_label),
                    (bottom_arrays, bottom_labels, bottom_cmap, bottom_vmin, bottom_vmax, bottom_colorbar_label),
                )
            ):
                row_idx = 2 * chunk_idx + local_row_idx
                handle = None
                for col_idx, array in enumerate(arrays):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    handle = self.draw_map(
                        ax,
                        array,
                        cmap,
                        vmin,
                        vmax,
                        labels[col_idx],
                        fill_panel=(local_row_idx == 1 and bottom_fill_panel),
                    ) or handle
                cax = fig.add_subplot(gs[row_idx, n_cols])
                if handle is not None and vmin is not None and vmax is not None:
                    cb = fig.colorbar(self.colorbar_mappable(handle, cmap, vmin, vmax), cax=cax)
                    cb.set_label(cb_label, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax.axis("off")
        self.save_figure(fig, out_base)

    # =========================
    # (9) TBL/TWCF u-v 光流图
    # =========================
    def plot_tbl_twcf_flow_uv(self, group: GroupContext) -> None:
        """图九：TBL/TWCF 每行一个实验，横向展示 u 与 v 光流图；TBL 第一行加入 GT。"""

        for bundle in self.limited_bundles(group, "flow"):
            u_maps: dict[str, np.ndarray] = {}
            v_maps: dict[str, np.ndarray] = {}
            gt_u = None
            gt_v = None
            for exp_key, sample_dir in bundle.sample_dirs.items():
                maps = self.load_flow_value_maps(sample_dir)
                gt_u = gt_u if gt_u is not None else maps.get("gt_u")
                gt_v = gt_v if gt_v is not None else maps.get("gt_v")
                if "pred_u" in maps:
                    u_maps[exp_key] = maps["pred_u"]
                if "pred_v" in maps:
                    v_maps[exp_key] = maps["pred_v"]
            exp_keys = [key for key in self.legend_order_keys() if key in u_maps or key in v_maps]
            rows: list[tuple[str, np.ndarray | None, np.ndarray | None, str | None]] = []
            if getattr(self.cfg, "TBL_FLOW_UV_INCLUDE_GT", True) and (gt_u is not None or gt_v is not None):
                rows.append((self.cfg.FLOW_GT_PANEL_LABEL, gt_u, gt_v, None))
            rows.extend((self.experiment_label(exp_key), u_maps.get(exp_key), v_maps.get(exp_key), exp_key) for exp_key in exp_keys)
            if not rows:
                continue
            all_arrays = [row[1] for row in rows if row[1] is not None] + [row[2] for row in rows if row[2] is not None]
            vmin, vmax = self.resolve_color_limit(all_arrays, self.cfg.FLOW_VALUE_COLORBAR_LIMIT)
            fig = plt.figure(
                figsize=(
                    float(getattr(self.cfg, "TBL_FLOW_UV_FIG_WIDTH", 6.0)),
                    float(getattr(self.cfg, "TBL_FLOW_UV_ROW_HEIGHT", 1.55)) * len(rows),
                )
            )
            gs = fig.add_gridspec(len(rows), 3, width_ratios=[1, 1, 0.07], hspace=0.08, wspace=0.05)
            for row_idx, (row_label, u_array, v_array, _exp_key) in enumerate(rows):
                show_component_label = (
                    row_idx == 0 or not getattr(self.cfg, "TBL_FLOW_UV_COMPONENT_LABELS_FIRST_ROW_ONLY", True)
                )
                ax_u = fig.add_subplot(gs[row_idx, 0])
                row_handle = self.draw_map(
                    ax_u,
                    u_array,
                    self.cfg.IMAGE_CMAP,
                    vmin,
                    vmax,
                    "u" if show_component_label else "",
                )
                ax_u.text(
                    -0.08,
                    0.5,
                    row_label,
                    transform=ax_u.transAxes,
                    ha="right",
                    va="center",
                    fontsize=self.cfg.PANEL_LABEL_SIZE,
                )
                ax_v = fig.add_subplot(gs[row_idx, 1])
                row_handle = self.draw_map(
                    ax_v,
                    v_array,
                    self.cfg.IMAGE_CMAP,
                    vmin,
                    vmax,
                    "v" if show_component_label else "",
                ) or row_handle
                cax = fig.add_subplot(gs[row_idx, 2])
                if row_handle is not None:
                    cb = fig.colorbar(row_handle, cax=cax)
                    cb.set_label(self.cfg.FLOW_VALUE_COLORBAR_LABEL, fontsize=self.cfg.COLORBAR_LABEL_SIZE)
                else:
                    cax.axis("off")
            out_dir = self.output_dir(
                self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                group.class_name,
                group.split_name,
                group.category_name,
                bundle.sample_name,
            )
            self.save_figure(fig, out_dir / f"{normalize_name(group.category_name)}_flow_uv_panel")

    def discover_tbl_profile_sample_dirs(self, category_dir: Path) -> dict[str, Path]:
        """发现 TBL 已保存的剖面分析样本目录：category/profile_analysis/sample_xxxx。"""

        profile_root = category_dir / getattr(self.cfg, "TBL_PROFILE_DIR_NAME", "profile_analysis")
        if not profile_root.exists():
            return {}
        sample_dirs: dict[str, Path] = {}
        for child in profile_root.iterdir():
            if not child.is_dir():
                continue
            has_profile = False
            for component in getattr(self.cfg, "TBL_PROFILE_COMPONENTS", ("u", "v")):
                pred_name = self.tbl_profile_file_name("pred", component)
                gt_name = self.tbl_profile_file_name("gt", component)
                if (child / pred_name).exists() or (child / gt_name).exists():
                    has_profile = True
                    break
            if has_profile:
                sample_dirs[child.name] = child
        return sample_dirs

    def tbl_profile_file_name(self, key: str, component: str) -> str:
        """按全局模板生成 TBL profile 文件名，兼容后续修改 u/v 文件名。"""

        template = getattr(self.cfg, "TBL_PROFILE_FILE_NAMES", {}).get(key, "")
        return str(template).format(component=component)

    def bundle_tbl_profile_samples(self, group: GroupContext) -> list[SampleBundle]:
        """把同名 TBL profile sample 在各实验中的目录合并，供剖面叠加图统一读取。"""

        per_experiment = {
            exp_key: self.discover_tbl_profile_sample_dirs(category_dir)
            for exp_key, category_dir in group.experiment_dirs.items()
        }
        names: set[str] = set()
        for sample_map in per_experiment.values():
            names.update(sample_map.keys())
        if self.cfg.SAMPLE_FILTER:
            allowed = {normalize_name(v) for v in self.cfg.SAMPLE_FILTER}
            names = {name for name in names if normalize_name(name) in allowed}

        bundles: list[SampleBundle] = []
        for name in sorted(names):
            dirs = {
                exp_key: sample_map[name]
                for exp_key, sample_map in per_experiment.items()
                if name in sample_map
            }
            if dirs:
                bundles.append(SampleBundle(sample_name=name, sample_dirs=dirs))

        limit = self.cfg.MAX_SAMPLE_COMPOSITES_PER_CATEGORY
        if limit is not None:
            return bundles[: int(limit)]
        return bundles

    def normalize_profile_array(self, array: np.ndarray | None, y_positions: np.ndarray | None) -> np.ndarray | None:
        """
        将 TBL profile 统一成 region x y_points。
        历史文件通常是 (3, 200)，但这里仍兼容 (200, 3) 或单剖面一维数组，避免不同保存脚本造成维度错位。
        """

        if array is None:
            return None
        arr = np.asarray(array, dtype=np.float64).squeeze()
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        y_size = int(np.asarray(y_positions).size) if y_positions is not None else 0
        if y_size and arr.shape[1] == y_size:
            return arr
        if y_size and arr.shape[0] == y_size:
            return arr.T
        # 没有 y 坐标时，TBL 通常是“剖面数 x 采样点”；若第一维远大于第二维，则更可能是转置格式。
        return arr.T if arr.shape[0] > arr.shape[1] else arr

    def load_tbl_profile(
        self, profile_dir: Path, component: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """读取单个实验的 TBL u/v profile：pred、gt、y 坐标和剖面列位置。"""

        y_positions = load_npy(profile_dir / self.tbl_profile_file_name("y", component))
        columns = load_npy(profile_dir / self.tbl_profile_file_name("columns", component))
        pred = self.normalize_profile_array(load_npy(profile_dir / self.tbl_profile_file_name("pred", component)), y_positions)
        gt = self.normalize_profile_array(load_npy(profile_dir / self.tbl_profile_file_name("gt", component)), y_positions)
        y = None
        if y_positions is not None:
            y = np.asarray(y_positions, dtype=np.float64).reshape(-1)
        if y is None:
            point_count = 0
            if pred is not None:
                point_count = pred.shape[1]
            elif gt is not None:
                point_count = gt.shape[1]
            if point_count:
                y = np.arange(point_count, dtype=np.float64)
        return pred, gt, y, columns

    def load_tbl_profile_top_map_from_dir(
        self, profile_dir: Path, component: str, y_positions: np.ndarray | None
    ) -> np.ndarray | None:
        """从 profile_analysis 目录读取顶部 GT 流场，确保虚线和剖面列坐标使用同一坐标系。"""

        # 优先读取原 evaluate 代码专门保存的 *_gt_profile_view.npy。
        # 这个文件已经裁到 profile_y_positions 的高度，宽度仍是 TBL full-frame 宽度，
        # 因此 profile_columns.npy 可以直接画在这张图上，不会发生坐标缩放错位。
        file_map = getattr(self.cfg, "TBL_PROFILE_TOP_FIELD_FILE_NAMES", {})
        top_name = str(file_map.get(component, f"{component}_gt_profile_view.npy"))
        top_map = load_npy(profile_dir / top_name)
        if top_map is not None:
            return ensure_2d_image(top_map)

        # 兼容旧数据：如果没有 *_gt_profile_view.npy，则读取完整 *_gt.npy，
        # 再按 y_positions 的长度裁到同样的有效边界层高度。
        full_map = load_npy(profile_dir / f"{component}_gt.npy")
        if full_map is None:
            return None
        arr = ensure_2d_image(full_map)
        if arr.ndim != 2:
            return None
        if y_positions is not None and np.asarray(y_positions).size:
            height = min(arr.shape[0], int(np.asarray(y_positions).size))
            arr = arr[:height, :]
        return arr

    def load_tbl_profile_gt_map(self, group: GroupContext, sample_name: str, component: str) -> np.ndarray | None:
        """兜底读取 TBL 剖面图顶部 GT U/V 底图；仅在 profile_analysis 无底图时使用。"""

        for exp_key in self.legend_order_keys():
            category_dir = group.experiment_dirs.get(exp_key)
            if category_dir is None:
                continue
            sample_dirs = self.discover_sample_dirs(category_dir, "flow")
            sample_dir = sample_dirs.get(sample_name)
            if sample_dir is None:
                continue
            maps = self.load_flow_value_maps(sample_dir)
            gt_map = maps.get(f"gt_{component}")
            if gt_map is not None:
                return gt_map
        return None

    def resolve_tbl_profile_plot_columns(
        self, profile_columns: np.ndarray | None, top_width: int, region_count: int
    ) -> np.ndarray | None:
        """把 profile_columns.npy 转换成顶部 GT 流场的 x 坐标，避免不同宽度底图导致虚线偏移。"""

        if top_width <= 0 or profile_columns is None:
            return None
        columns = np.asarray(profile_columns, dtype=np.float64).reshape(-1)
        columns = columns[np.isfinite(columns)]
        if columns.size == 0:
            return None
        columns = columns[:region_count]
        if np.nanmax(columns) < top_width:
            return np.clip(columns, 0, top_width - 1)

        # 如果顶部图不是原 profile 视图尺寸，就使用全局比例重新定位三条线。
        # 旧逻辑会按 max(column) 缩放，导致 Turbulent 被错误放到最右端；这里改成原始比例。
        ratios = np.asarray(getattr(self.cfg, "TBL_PROFILE_COLUMN_RATIOS", ()), dtype=np.float64).reshape(-1)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size >= columns.size:
            return np.clip(ratios[: columns.size] * float(top_width - 1), 0, top_width - 1)

        reference_width = max(float(np.nanmax(columns)) / 0.83, float(np.nanmax(columns)) + 1.0)
        return np.clip(columns / reference_width * float(top_width - 1), 0, top_width - 1)

    def plot_tbl_profile_overlays(self, group: GroupContext) -> None:
        """TBL 补充图：按图三样式绘制顶部 GT 场和下方三段剖面对比曲线。"""

        for bundle in self.bundle_tbl_profile_samples(group):
            for component in getattr(self.cfg, "TBL_PROFILE_COMPONENTS", ("u", "v")):
                pred_profiles: dict[str, np.ndarray] = {}
                gt_profile: np.ndarray | None = None
                y_positions: np.ndarray | None = None
                profile_columns: np.ndarray | None = None
                top_gt_map: np.ndarray | None = None

                for exp_key, profile_dir in bundle.sample_dirs.items():
                    pred, gt, y, columns = self.load_tbl_profile(profile_dir, component)
                    if pred is not None:
                        pred_profiles[exp_key] = pred
                    if gt_profile is None and gt is not None:
                        gt_profile = gt
                    if y_positions is None and y is not None:
                        y_positions = y
                    if profile_columns is None and columns is not None:
                        profile_columns = np.asarray(columns).reshape(-1)
                    if top_gt_map is None:
                        top_gt_map = self.load_tbl_profile_top_map_from_dir(profile_dir, component, y)

                if not pred_profiles and gt_profile is None:
                    continue
                n_regions = 0
                if gt_profile is not None:
                    n_regions = max(n_regions, gt_profile.shape[0])
                for profile in pred_profiles.values():
                    n_regions = max(n_regions, profile.shape[0])
                if n_regions <= 0:
                    continue
                if y_positions is None:
                    point_count = gt_profile.shape[1] if gt_profile is not None else next(iter(pred_profiles.values())).shape[1]
                    y_positions = np.arange(point_count, dtype=np.float64)

                self.plot_tbl_profile_overlay_figure(
                    group,
                    bundle.sample_name,
                    component,
                    pred_profiles,
                    gt_profile,
                    y_positions,
                    profile_columns,
                    top_gt_map,
                    exclude_keys=(),
                    suffix="",
                )
                extra_exclude = tuple(getattr(self.cfg, "TBL_PROFILE_EXTRA_EXCLUDE_EXPERIMENT_KEYS", ()))
                if extra_exclude:
                    self.plot_tbl_profile_overlay_figure(
                        group,
                        bundle.sample_name,
                        component,
                        pred_profiles,
                        gt_profile,
                        y_positions,
                        profile_columns,
                        top_gt_map,
                        exclude_keys=extra_exclude,
                        suffix=str(getattr(self.cfg, "TBL_PROFILE_EXTRA_SUFFIX", "_filtered")),
                    )

    def plot_tbl_profile_overlay_figure(
        self,
        group: GroupContext,
        sample_name: str,
        component: str,
        pred_profiles: dict[str, np.ndarray],
        gt_profile: np.ndarray | None,
        y_positions: np.ndarray,
        profile_columns: np.ndarray | None,
        top_gt_map: np.ndarray | None,
        exclude_keys: tuple[str, ...],
        suffix: str,
    ) -> None:
        """绘制一张 TBL profile 对比图：顶部 GT 底图，下方三段剖面曲线，图例放在中间子图内。"""

        exp_keys = [
            exp_key
            for exp_key in self.legend_order_keys()
            if exp_key in pred_profiles and exp_key not in set(exclude_keys)
        ]
        if not exp_keys and gt_profile is None:
            return
        n_regions = max([gt_profile.shape[0] if gt_profile is not None else 0] + [pred_profiles[k].shape[0] for k in exp_keys])
        fig_width = max(6.5, float(getattr(self.cfg, "TBL_PROFILE_FIG_WIDTH_PER_REGION", 3.4)) * n_regions)
        fig_height = float(getattr(self.cfg, "TBL_PROFILE_FIG_HEIGHT", 8.8))
        fig = plt.figure(figsize=(fig_width, fig_height))
        # 增大左边距，避免 previous/next 行标签和灰度直方图 y 轴 label 挤在一起。
        fig.subplots_adjust(
            left=float(getattr(self.cfg, "PARTICLE_STATS_SUBPLOTS_LEFT", 0.075)),
            right=float(getattr(self.cfg, "PARTICLE_STATS_SUBPLOTS_RIGHT", 0.985)),
        )
        gs = fig.add_gridspec(
            3,
            n_regions,
            height_ratios=[
                float(getattr(self.cfg, "TBL_PROFILE_TOP_HEIGHT_RATIO", 0.85)),
                float(getattr(self.cfg, "TBL_PROFILE_COLORBAR_HEIGHT_RATIO", 0.10)),
                float(getattr(self.cfg, "TBL_PROFILE_CURVE_HEIGHT_RATIO", 2.30)),
            ],
            hspace=float(getattr(self.cfg, "TBL_PROFILE_HSPACE", 0.58)),
            wspace=float(getattr(self.cfg, "TBL_PROFILE_WSPACE", 0.18)),
        )

        top_ax = fig.add_subplot(gs[0, :])
        gt_map = top_gt_map if top_gt_map is not None else self.load_tbl_profile_gt_map(group, sample_name, component)
        top_handle = None
        top_vmin = None
        top_vmax = None
        if gt_map is not None:
            top_arr = ensure_2d_image(gt_map)
            top_vmin, top_vmax = self.resolve_color_limit([top_arr], self.cfg.FLOW_VALUE_COLORBAR_LIMIT)
            top_handle = top_ax.imshow(
                top_arr,
                cmap=self.cfg.IMAGE_CMAP,
                vmin=top_vmin,
                vmax=top_vmax,
                origin="lower",
                aspect="auto",
            )
            # 顶部图按论文图样式使用居中标题；左上角 GT U/GT V 标签仍保留为全局可配文本。
            top_title = getattr(self.cfg, "TBL_PROFILE_TOP_TITLES", {}).get(component, "")
            if top_title:
                top_ax.set_title(top_title, fontsize=self.cfg.AXIS_LABEL_SIZE, pad=2)
            self.panel_text(
                top_ax,
                getattr(self.cfg, "TBL_PROFILE_TOP_LABELS", {}).get(component, f"GT {component.upper()}"),
            )
            top_ax.set_xticks([])
            top_ax.set_yticks([])
        else:
            top_ax.axis("off")
        region_labels = tuple(getattr(self.cfg, "TBL_PROFILE_REGION_LABELS", ()))
        if gt_map is not None and profile_columns is not None:
            top_arr = ensure_2d_image(gt_map)
            top_width = top_arr.shape[1] if top_arr.ndim >= 2 else 0
            plot_columns = self.resolve_tbl_profile_plot_columns(profile_columns, top_width, n_regions)
            if plot_columns is not None:
                for idx, x_pos in enumerate(plot_columns[:n_regions]):
                    top_ax.axvline(float(x_pos), color="red", linestyle="--", linewidth=1.2)
                    label = region_labels[idx] if idx < len(region_labels) else f"profile {idx + 1}"
                    top_ax.text(
                        float(x_pos),
                        0.96,
                        label,
                        transform=top_ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        color="red",
                        fontsize=self.cfg.TICK_LABEL_SIZE,
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
                    )
        # 中间行按用户要求拆成左右两块：左边放顶部 GT 流场色条，右边放整张剖面图图例。
        # 图例不再放入下方三张剖面子图内，避免遮挡曲线或区域标题。
        colorbar_row = gs[1, :].subgridspec(
            1,
            2,
            width_ratios=tuple(getattr(self.cfg, "TBL_PROFILE_COLORBAR_LEGEND_WIDTH_RATIOS", (1.15, 0.85))),
            wspace=float(getattr(self.cfg, "TBL_PROFILE_COLORBAR_LEGEND_WSPACE", 0.18)),
        )
        cax = fig.add_subplot(colorbar_row[0, 0])
        legend_ax = fig.add_subplot(colorbar_row[0, 1])
        legend_ax.axis("off")
        if top_handle is not None and gt_map is not None and ensure_2d_image(gt_map).ndim == 2:
            cb = fig.colorbar(top_handle, cax=cax, orientation="horizontal")
            # 色条 label 放到上方，避免和下面三张剖面图的区域标题/坐标区域挤在一起。
            cb.ax.xaxis.set_label_position(getattr(self.cfg, "TBL_PROFILE_COLORBAR_LABEL_POSITION", "top"))
            cb.set_label(
                self.cfg.FLOW_VALUE_COLORBAR_LABEL,
                fontsize=self.cfg.COLORBAR_LABEL_SIZE,
                labelpad=float(getattr(self.cfg, "TBL_PROFILE_COLORBAR_LABEL_PAD", 7)),
            )
        else:
            cax.axis("off")

        x_label_map = getattr(self.cfg, "TBL_PROFILE_X_LABELS", {})
        x_label = x_label_map.get(component, f"{component} displacement [px]")
        axes = [fig.add_subplot(gs[2, idx]) for idx in range(n_regions)]
        for region_idx, ax in enumerate(axes):
            y = y_positions
            if gt_profile is not None and region_idx < gt_profile.shape[0]:
                gt_values = gt_profile[region_idx]
                y_gt = y[: gt_values.size] if y.size >= gt_values.size else np.arange(gt_values.size)
                ax.plot(
                    gt_values,
                    y_gt,
                    color=getattr(self.cfg, "TBL_PROFILE_GT_COLOR", "#444444"),
                    linestyle=getattr(self.cfg, "TBL_PROFILE_GT_LINESTYLE", "--"),
                    linewidth=float(getattr(self.cfg, "TBL_PROFILE_GT_LINE_WIDTH", 1.4)),
                    label=getattr(self.cfg, "TBL_PROFILE_GT_LABEL", "GT"),
                )
            for exp_key in exp_keys:
                profile = pred_profiles.get(exp_key)
                if profile is None or region_idx >= profile.shape[0]:
                    continue
                values = profile[region_idx]
                y_pred = y[: values.size] if y.size >= values.size else np.arange(values.size)
                ax.plot(
                    values,
                    y_pred,
                    # TBL 剖面图使用普通实验图例颜色，不跟误差直方图专用配色走；
                    # 这样 bicubic-hs 和 ESRuRAFT-PIV 会严格使用 global_class.py 中 EXPERIMENT_COLORS 的设置。
                    color=self.experiment_color(exp_key),
                    linewidth=float(getattr(self.cfg, "TBL_PROFILE_PRED_LINE_WIDTH", 1.25)),
                    alpha=float(getattr(self.cfg, "TBL_PROFILE_ALPHA", 0.95)),
                    label=self.experiment_label(exp_key),
                )
            panel_label = region_labels[region_idx] if region_idx < len(region_labels) else f"profile {region_idx + 1}"
            # 下方三张剖面图使用居中标题，贴近原始论文风格 profile_compare 图。
            ax.set_title(panel_label, fontsize=self.cfg.AXIS_LABEL_SIZE, fontweight="bold", pad=2)
            ax.set_xlabel(x_label)
            if region_idx == 0:
                ax.set_ylabel(getattr(self.cfg, "TBL_PROFILE_Y_LABEL", "y [px]"))
            ax.grid(True, alpha=float(getattr(self.cfg, "TBL_PROFILE_GRID_ALPHA", 0.25)), linewidth=0.5)
            self.apply_axis_limits(
                ax,
                getattr(self.cfg, "TBL_PROFILE_X_MIN", None),
                getattr(self.cfg, "TBL_PROFILE_X_MAX", None),
                getattr(self.cfg, "TBL_PROFILE_Y_MIN", None),
                getattr(self.cfg, "TBL_PROFILE_Y_MAX", None),
            )

        legend_handles = []
        legend_labels = []
        seen_labels = set()
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in seen_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
                    seen_labels.add(label)
        if legend_handles:
            legend_ax.legend(
                legend_handles,
                legend_labels,
                loc=getattr(self.cfg, "TBL_PROFILE_LEGEND_LOC", "center"),
                frameon=True,
                fontsize=self.cfg.LEGEND_FONT_SIZE,
                ncol=min(len(legend_labels), int(getattr(self.cfg, "TBL_PROFILE_LEGEND_NCOL", 2))),
            )

        out_dir = self.output_dir(
            self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
            group.class_name,
            group.split_name,
            group.category_name,
            sample_name,
        )
        out_base = out_dir / f"tbl_{component}_profile_overlay{suffix}"
        self.save_npy(
            out_base.with_suffix(".npy"),
            {
                "y": y_positions,
                "columns": profile_columns,
                "top_gt_map": top_gt_map,
                "gt": gt_profile,
                "pred": {key: pred_profiles[key] for key in exp_keys if key in pred_profiles},
                "excluded": exclude_keys,
            },
        )
        self.save_figure(fig, out_base)

    # =========================
    # (10) 颗粒统计组合图
    # =========================
    def plot_particle_stats_composites(self, group: GroupContext, metrics_only: bool = False) -> None:
        """图十：颗粒统计图拆成图像对比图和更大的指标统计图。"""

        for bundle in self.limited_bundles(group, "particle_stats"):
            exp_keys = [key for key in group.experiment_keys if key in bundle.sample_dirs]
            if not exp_keys:
                continue
            out_dir = self.output_dir(
                self.cfg.COMPOSITE_OUTPUT_DIR_NAME,
                group.class_name,
                group.split_name,
                group.category_name,
                bundle.sample_name,
            )
            is_tbl = normalize_name(group.category_name) == "tbl"
            if not metrics_only:
                self.plot_particle_stats_image_composite(bundle, exp_keys, out_dir, crop=False, is_tbl=is_tbl)
            self.plot_particle_stats_metric_composite(bundle, exp_keys, out_dir, crop=False)
            # TBL 的 crop 颗粒统计与原图统计都要保留：
            # full-frame 用原始 npy/csv，crop 版优先读取 *_crop*.npy，SR/GT 图没有 crop npy 时再按同一 crop 框裁。
            if is_tbl and getattr(self.cfg, "TBL_PARTICLE_CROP_ENABLED", True):
                if not metrics_only:
                    self.plot_particle_stats_image_composite(bundle, exp_keys, out_dir, crop=True, is_tbl=is_tbl)
                self.plot_particle_stats_metric_composite(bundle, exp_keys, out_dir, crop=True)

    def particle_stats_metric_config(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """集中返回颗粒统计条形图的指标 key 和图上显示 label。"""

        # 按用户要求，颗粒统计拆分后的条形统计图只保留 count 和 particle pixels；
        # IoU/precision/recall/F1 不再画入这张图，避免产生多余空行和过密子图。
        metrics = ("count", "pixels")
        metric_labels = (
            self.cfg.PARTICLE_STAT_COUNT_LABEL,
            self.cfg.PARTICLE_STAT_PIXEL_LABEL,
        )
        return metrics, metric_labels

    def plot_particle_stats_image_composite(
        self, bundle: SampleBundle, exp_keys: list[str], out_dir: Path, crop: bool = False, is_tbl: bool = False
    ) -> None:
        """只绘制 previous/next 的 GT、各实验 SR 图和阈值图；crop=True 时读取 TBL crop 数据。"""

        if is_tbl and not crop and getattr(self.cfg, "TBL_PARTICLE_STATS_IMAGE_VERTICAL_LAYOUT", True):
            self.plot_particle_stats_image_composite_vertical(bundle, exp_keys, out_dir)
            return

        n_cols = 1 + len(exp_keys)
        # 颗粒图/阈值图只需要展示图像本身，不能复用条形统计图的宽列参数；
        # 这里使用单独的紧凑画布，避免 crop 阈值图在横向多列时被大片空白隔开。
        fig_width = max(7.5, float(getattr(self.cfg, "PARTICLE_STATS_IMAGE_WIDTH_PER_COL", 1.35)) * n_cols)
        fig_height = float(getattr(self.cfg, "PARTICLE_STATS_IMAGE_COMPACT_FIG_HEIGHT", self.cfg.PARTICLE_STATS_IMAGE_FIG_HEIGHT))
        fig = plt.figure(figsize=(fig_width, fig_height))
        height_ratios = [
            self.cfg.PARTICLE_STATS_IMAGE_ROW_RATIO,
            self.cfg.PARTICLE_STATS_IMAGE_ROW_RATIO,
            float(getattr(self.cfg, "PARTICLE_STATS_IMAGE_COMPACT_BLOCK_GAP_RATIO", self.cfg.PARTICLE_STATS_BLOCK_GAP_RATIO)),
            self.cfg.PARTICLE_STATS_IMAGE_ROW_RATIO,
            self.cfg.PARTICLE_STATS_IMAGE_ROW_RATIO,
        ]
        gs = fig.add_gridspec(
            5,
            n_cols,
            height_ratios=height_ratios,
            hspace=float(getattr(self.cfg, "PARTICLE_STATS_IMAGE_COMPACT_HSPACE", self.cfg.PARTICLE_STATS_HSPACE)),
            wspace=float(getattr(self.cfg, "PARTICLE_STATS_IMAGE_COMPACT_WSPACE", self.cfg.PARTICLE_STATS_WSPACE)),
        )
        gap_ax = fig.add_subplot(gs[2, :])
        gap_ax.axis("off")

        for block_idx, time_name in enumerate(("previous", "next")):
            row0 = 0 if block_idx == 0 else 3
            gt = self.first_available_particle(bundle, time_name, "gt", crop=crop)
            gt_binary = self.first_available_particle(bundle, time_name, "gt_binary", crop=crop)
            sr_maps = {
                exp_key: self.load_particle_array_mode(sample_dir, time_name, "sr", crop=crop)
                for exp_key, sample_dir in bundle.sample_dirs.items()
            }
            sr_binary_maps = {
                exp_key: self.load_particle_array_mode(sample_dir, time_name, "sr_binary", crop=crop)
                for exp_key, sample_dir in bundle.sample_dirs.items()
            }
            image_vmin, image_vmax = self.row_limit([gt] + list(sr_maps.values()), self.cfg.PARTICLE_VALUE_COLORBAR_LIMIT)
            binary_vmin, binary_vmax = 0.0, 1.0

            first_row_arrays = [gt] + [sr_maps.get(exp_key) for exp_key in exp_keys]
            first_row_labels = [self.cfg.GT_PANEL_LABEL] + [self.experiment_label(exp_key) for exp_key in exp_keys]
            second_row_arrays = [gt_binary] + [sr_binary_maps.get(exp_key) for exp_key in exp_keys]
            second_row_labels = [self.cfg.GT_PANEL_LABEL] + [self.experiment_label(exp_key) for exp_key in exp_keys]
            label_size = float(getattr(self.cfg, "PARTICLE_BINARY_PANEL_LABEL_SIZE", self.cfg.PANEL_LABEL_SIZE))
            for col_idx in range(n_cols):
                ax = fig.add_subplot(gs[row0, col_idx])
                self.draw_map(
                    ax,
                    first_row_arrays[col_idx] if col_idx < len(first_row_arrays) else None,
                    self.cfg.IMAGE_CMAP,
                    image_vmin,
                    image_vmax,
                    first_row_labels[col_idx] if col_idx < len(first_row_labels) else "",
                    label_fontsize=label_size,
                )
                if col_idx == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        self.cfg.PREVIOUS_ROW_LABEL if time_name == "previous" else self.cfg.NEXT_ROW_LABEL,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        rotation=90,
                        fontsize=self.cfg.PANEL_LABEL_SIZE,
                    )
                ax = fig.add_subplot(gs[row0 + 1, col_idx])
                self.draw_map(
                    ax,
                    second_row_arrays[col_idx] if col_idx < len(second_row_arrays) else None,
                    self.cfg.BINARY_CMAP,
                    binary_vmin,
                    binary_vmax,
                    second_row_labels[col_idx] if col_idx < len(second_row_labels) else "",
                    label_fontsize=label_size,
                )

        suffix = getattr(self.cfg, "TBL_PARTICLE_CROP_OUTPUT_SUFFIX", "_crop") if crop else ""
        # 保留原文件名，但内容改为纯图像/阈值对比；统计行已拆到 particle_binary_stats_metrics_composite。
        self.save_figure(fig, out_dir / f"particle_binary_stats_composite{suffix}")

    def plot_particle_stats_image_composite_vertical(
        self, bundle: SampleBundle, exp_keys: list[str], out_dir: Path
    ) -> None:
        """TBL full-frame 颗粒图/阈值图纵向两列排版：左颗粒图，右阈值图。"""

        rows: list[tuple[str, str, np.ndarray | None, np.ndarray | None]] = []
        for time_name in ("previous", "next"):
            gt = self.first_available_particle(bundle, time_name, "gt", crop=False)
            gt_binary = self.first_available_particle(bundle, time_name, "gt_binary", crop=False)
            rows.append((time_name, self.cfg.GT_PANEL_LABEL, gt, gt_binary))
            for exp_key in exp_keys:
                sample_dir = bundle.sample_dirs.get(exp_key)
                sr = self.load_particle_array_mode(sample_dir, time_name, "sr", crop=False) if sample_dir is not None else None
                sr_binary = (
                    self.load_particle_array_mode(sample_dir, time_name, "sr_binary", crop=False)
                    if sample_dir is not None
                    else None
                )
                rows.append((time_name, self.experiment_label(exp_key), sr, sr_binary))
            if time_name == "previous":
                rows.append(("gap", "", None, None))

        row_count = len(rows)
        fig_width = float(getattr(self.cfg, "TBL_STATS_IMAGE_VERTICAL_FIG_WIDTH", 8.6))
        row_height = float(getattr(self.cfg, "TBL_STATS_IMAGE_VERTICAL_ROW_HEIGHT", 1.15))
        fig = plt.figure(figsize=(fig_width, max(3.0, row_height * row_count)))
        height_ratios = [0.22 if time_name == "gap" else 1.0 for time_name, _, _, _ in rows]
        gs = fig.add_gridspec(
            row_count,
            2,
            height_ratios=height_ratios,
            hspace=float(getattr(self.cfg, "TBL_STATS_IMAGE_VERTICAL_HSPACE", 0.06)),
            wspace=float(getattr(self.cfg, "TBL_STATS_IMAGE_VERTICAL_WSPACE", 0.05)),
        )
        image_arrays = [array for time_name, _, array, _ in rows if time_name != "gap"]
        image_vmin, image_vmax = self.row_limit(image_arrays, self.cfg.PARTICLE_VALUE_COLORBAR_LIMIT)
        label_size = float(getattr(self.cfg, "PARTICLE_BINARY_PANEL_LABEL_SIZE", self.cfg.PANEL_LABEL_SIZE))
        for row_idx, (time_name, label, image_array, binary_array) in enumerate(rows):
            if time_name == "gap":
                ax = fig.add_subplot(gs[row_idx, :])
                ax.axis("off")
                continue
            ax = fig.add_subplot(gs[row_idx, 0])
            self.draw_map(
                ax,
                image_array,
                self.cfg.IMAGE_CMAP,
                image_vmin,
                image_vmax,
                label,
                fill_panel=True,
                label_fontsize=label_size,
            )
            if label == self.cfg.GT_PANEL_LABEL:
                ax.text(
                    -0.02,
                    0.5,
                    self.cfg.PREVIOUS_ROW_LABEL if time_name == "previous" else self.cfg.NEXT_ROW_LABEL,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=self.cfg.PANEL_LABEL_SIZE,
                )
            ax = fig.add_subplot(gs[row_idx, 1])
            self.draw_map(
                ax,
                binary_array,
                self.cfg.BINARY_CMAP,
                0.0,
                1.0,
                f"{label} binary",
                fill_panel=True,
                label_fontsize=label_size,
            )

        self.save_figure(fig, out_dir / "particle_binary_stats_composite")

    def plot_particle_stats_metric_composite(
        self, bundle: SampleBundle, exp_keys: list[str], out_dir: Path, crop: bool = False
    ) -> None:
        """单独绘制 GT 灰度直方图和统计条形图；crop=True 时使用 TBL 已保存 crop stats/hist。"""

        metrics, metric_labels = self.particle_stats_metric_config()
        # 现在只保留灰度直方图、count 和 particle pixels，previous/next 各占一行；
        # 去掉原来为 IoU/precision/recall/F1 预留的第二行和中间空行。
        n_cols = 1 + len(metrics)
        fig_width = max(12.0, self.cfg.PARTICLE_STATS_FIG_WIDTH_PER_COL * n_cols * 1.45)
        fig_height = max(5.6, float(self.cfg.PARTICLE_STATS_METRIC_FIG_HEIGHT) * 0.55)
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            2,
            n_cols,
            height_ratios=[self.cfg.PARTICLE_STATS_CHART_ROW_RATIO, self.cfg.PARTICLE_STATS_CHART_ROW_RATIO],
            hspace=self.cfg.PARTICLE_STATS_HSPACE,
            wspace=self.cfg.PARTICLE_STATS_WSPACE,
        )
        patch_cls = ensure_matplotlib().matplotlib.patches.Patch
        legend_handles = [
            patch_cls(facecolor=self.particle_stats_bar_color(key), edgecolor=self.cfg.PARTICLE_STATS_BAR_EDGE_COLOR)
            for key in ([None] + exp_keys)
        ]
        legend_labels = [self.cfg.GT_PANEL_LABEL] + [self.experiment_label(k) for k in exp_keys]

        for block_idx, time_name in enumerate(("previous", "next")):
            row_start = block_idx
            hist_ax = fig.add_subplot(gs[row_start, 0])
            self.draw_particle_gray_hist(hist_ax, bundle, time_name, crop=crop)
            hist_ax.text(
                float(getattr(self.cfg, "PARTICLE_STATS_ROW_LABEL_X", -0.23)),
                0.5,
                self.cfg.PREVIOUS_ROW_LABEL if time_name == "previous" else self.cfg.NEXT_ROW_LABEL,
                transform=hist_ax.transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=self.cfg.PANEL_LABEL_SIZE,
            )
            stats = {
                exp_key: self.load_particle_stats(sample_dir, time_name, crop=crop)
                for exp_key, sample_dir in bundle.sample_dirs.items()
            }
            gt_stats = self.load_particle_gt_stats(bundle, time_name, crop=crop)
            bar_labels = [self.cfg.GT_PANEL_LABEL] + [self.experiment_label(k) for k in exp_keys]
            bar_exp_keys = [None] + exp_keys
            bar_colors = [self.particle_stats_bar_color(key) for key in bar_exp_keys]

            for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                # count 和 particle pixels 与灰度直方图同行显示，不再保留其它指标的空行。
                plot_row = row_start
                plot_col = metric_idx + 1
                ax = fig.add_subplot(gs[plot_row, plot_col])
                values = np.asarray(
                    [gt_stats.get(metric, np.nan)]
                    + [stats.get(exp_key, {}).get(metric, np.nan) for exp_key in exp_keys],
                    dtype=np.float64,
                )
                x_pos = np.arange(len(bar_labels))
                finite_mask = np.isfinite(values)
                bars = ax.bar(
                    x_pos[finite_mask],
                    values[finite_mask],
                    color=[bar_colors[idx] for idx in np.where(finite_mask)[0]],
                    edgecolor=self.cfg.PARTICLE_STATS_BAR_EDGE_COLOR,
                    linewidth=0.6,
                    width=0.68,
                )
                # 在柱子顶部显示数值，GT 也参与标注，便于直接比较 GT 与各 SR 统计。
                for bar, value in zip(bars, values[finite_mask]):
                    ax.annotate(
                        self.format_particle_stats_value(float(value)),
                        xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=self.cfg.PARTICLE_STATS_VALUE_LABEL_SIZE,
                    )
                # 横轴长实验名已经移动到全局图例里，这里只保留柱子位置，不再显示 x tick label，避免图面拥挤。
                ax.set_xticks([])
                if np.any(finite_mask):
                    y_min = min(0.0, float(np.nanmin(values[finite_mask])))
                    y_max = float(np.nanmax(values[finite_mask]))
                    # y 轴顶部额外留白给图例和柱顶数值；留白比例放在 global_class.py，
                    # 后续如果图例仍然拥挤，只需要继续调大 PARTICLE_STATS_Y_PAD_RATIO。
                    pad_ratio = float(getattr(self.cfg, "PARTICLE_STATS_Y_PAD_RATIO", 0.45))
                    pad_min = float(getattr(self.cfg, "PARTICLE_STATS_Y_PAD_MIN", 1.0))
                    pad = (y_max - y_min) * pad_ratio if not math.isclose(y_max, y_min) else abs(y_max) * pad_ratio + pad_min
                    ax.set_ylim(y_min, y_max + pad)
                try:
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                except Exception:
                    pass
                ax.tick_params(axis="x", pad=1)
                ax.set_ylabel(self.cfg.PARTICLE_METRIC_Y_LABEL if metric not in ("count", "pixels") else self.cfg.PARTICLE_COUNT_Y_LABEL)
                self.panel_text(ax, label)
                ax.grid(True, alpha=0.18, linewidth=0.5)
                # 按用户要求，每张条形统计图内部都放图例；loc="best" 会自动寻找当前子图的空白区域。
                ax.legend(
                    legend_handles,
                    legend_labels,
                    loc=getattr(self.cfg, "PARTICLE_STATS_LEGEND_LOC", "best"),
                    frameon=False,
                    fontsize=float(getattr(self.cfg, "PARTICLE_STATS_LEGEND_FONT_SIZE", max(5, self.cfg.LEGEND_FONT_SIZE - 3))),
                    ncol=min(len(legend_labels), int(getattr(self.cfg, "PARTICLE_STATS_LEGEND_NCOL", 2))),
                )

        suffix = getattr(self.cfg, "TBL_PARTICLE_CROP_OUTPUT_SUFFIX", "_crop") if crop else ""
        self.save_figure(fig, out_dir / f"particle_binary_stats_metrics_composite{suffix}")

    def draw_particle_gray_hist(self, ax: plt.Axes, bundle: SampleBundle, time_name: str, crop: bool = False) -> None:
        """绘制 GT 灰度直方图，并标注阈值 T；TBL crop 版优先读取已保存的 crop hist。"""

        hist = None
        threshold = None
        for sample_dir in bundle.sample_dirs.values():
            # crop 统计图必须和局部 256x256 框一致，所以优先读 *_crop_hist.npy；
            # 如果历史目录没有 crop hist，则回退 full-frame hist，保证图仍能生成但 summary 中会保留缺文件警告入口。
            hist_path = self.particle_crop_path(sample_dir, time_name, "hist") if crop else self.particle_path(sample_dir, time_name, "hist")
            hist = load_npy(hist_path)
            if hist is None and crop:
                hist = load_npy(self.particle_path(sample_dir, time_name, "hist"))
            threshold = self.read_particle_threshold(sample_dir, time_name, crop=crop)
            if hist is not None:
                break
        xy = array_to_xy(hist) if hist is not None else None
        if xy is not None:
            x, y = xy
            ax.plot(x, y, color="#333333", linewidth=1.2)
        if threshold is not None:
            ax.axvline(threshold, color="#D55E00", linestyle="--", linewidth=1.2)
            # T=... 标注相对阈值线稍微右移，偏移量放在 global_class.py 中便于后续微调。
            text_dx = float(getattr(self.cfg, "PARTICLE_GRAY_HIST_THRESHOLD_TEXT_DX", 0.015))
            ax.text(
                threshold + text_dx,
                0.95,
                f"{self.cfg.THRESHOLD_LABEL}={threshold:.3g}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=self.cfg.TICK_LABEL_SIZE,
                color="#D55E00",
            )
        ax.set_xlabel(self.cfg.PARTICLE_GRAY_HIST_X_LABEL)
        ax.set_ylabel(self.cfg.PARTICLE_GRAY_HIST_Y_LABEL)
        ax.grid(True, alpha=0.18, linewidth=0.5)

    def read_particle_threshold(self, sample_dir: Path, time_name: str, crop: bool = False) -> float | None:
        """读取颗粒阈值；crop stats 若保存了 threshold 就用 crop，否则回退 full-frame threshold 文本。"""

        if crop:
            raw_stats = self.load_particle_stats_raw(sample_dir, time_name, crop=True)
            normalized_keys = {normalize_name(key): key for key in raw_stats.keys()}
            for alias in ("threshold", "threshold_value", "t"):
                real_key = normalized_keys.get(normalize_name(alias))
                if real_key is not None:
                    number = self.to_float(raw_stats.get(real_key))
                    if number is not None:
                        return number
        return self.read_threshold(sample_dir, time_name)

    def read_threshold(self, sample_dir: Path, time_name: str) -> float | None:
        path = self.particle_path(sample_dir, time_name, "threshold")
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8").strip()
            for token in text.replace("=", " ").replace(",", " ").split():
                try:
                    return float(token)
                except ValueError:
                    continue
        except Exception:
            return None
        return None

    def load_particle_stats(self, sample_dir: Path, time_name: str, crop: bool = False) -> dict[str, float]:
        """读取颗粒统计指标；TBL crop 版优先读已保存的 *_crop_stats.npy。"""

        if not crop:
            csv_path = self.particle_stats_csv_path(sample_dir, time_name)
            if csv_path.exists():
                stats = self.read_particle_stats_csv(csv_path)
                if stats:
                    return stats

        npy_path = self.particle_crop_path(sample_dir, time_name, "stats") if crop else self.particle_path(sample_dir, time_name, "stats")
        array = load_npy(npy_path)
        if array is None:
            return {}
        return self.parse_particle_stats_array(array)

    def load_particle_gt_stats(self, bundle: SampleBundle, time_name: str, crop: bool = False) -> dict[str, float]:
        """读取 GT 颗粒统计指标；crop=True 时使用 crop stats 的 gt_* 字段。"""

        fallback: dict[str, float] | None = None
        for sample_dir in bundle.sample_dirs.values():
            raw_stats = self.load_particle_stats_raw(sample_dir, time_name, crop=crop)
            mapped = self.map_gt_particle_stats_fields(raw_stats)
            # map_gt_particle_stats_fields 会固定加入 IoU/precision/recall/F1=1；
            # 因此这里不能只判断 mapped 非空，否则第一个没有 gt_count/gt_pixels 的文件会提前返回，
            # 导致 GT 的 count/pixels 缺失。优先返回带真实 GT 颗粒数/像素数的记录。
            if any(key in mapped for key in ("count", "pixels")):
                return mapped
            if mapped and fallback is None:
                fallback = mapped
        # 如果历史文件没有保存 GT count/pixels，仍给出 GT 自身二值重叠指标，count/pixels 留空为 NaN。
        return fallback or {
            "iou": float(self.cfg.PARTICLE_GT_SELF_METRIC_VALUE),
            "precision": float(self.cfg.PARTICLE_GT_SELF_METRIC_VALUE),
            "recall": float(self.cfg.PARTICLE_GT_SELF_METRIC_VALUE),
            "f1": float(self.cfg.PARTICLE_GT_SELF_METRIC_VALUE),
        }

    def load_particle_stats_raw(self, sample_dir: Path, time_name: str, crop: bool = False) -> dict:
        """读取未映射的颗粒统计字段；crop=True 时读取 TBL 局部统计 npy。"""

        if not crop:
            csv_path = self.particle_stats_csv_path(sample_dir, time_name)
            if csv_path.exists():
                raw_stats = self.read_particle_raw_stats_csv(csv_path)
                if raw_stats:
                    return raw_stats

        npy_path = self.particle_crop_path(sample_dir, time_name, "stats") if crop else self.particle_path(sample_dir, time_name, "stats")
        array = load_npy(npy_path)
        if array is None:
            return {}
        return self.parse_particle_raw_stats_array(array)

    def read_particle_stats_csv(self, path: Path) -> dict[str, float]:
        raw_stats = self.read_particle_raw_stats_csv(path)
        return self.map_stats_fields(raw_stats)

    def read_particle_raw_stats_csv(self, path: Path) -> dict:
        """读取原始颗粒统计 CSV；兼容 metric/value 两列表和普通一行多列表。"""

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                rows = list(csv.DictReader(file_obj))
        except Exception:
            return {}
        if not rows:
            return {}
        normalized_columns = {normalize_name(key): key for key in rows[0].keys()}
        metric_col = normalized_columns.get("metric")
        value_col = normalized_columns.get("value")
        if metric_col and value_col:
            # 颗粒统计 CSV 实际常保存成两列 metric/value：
            # 需要先还原成 {字段名: 数值}，再按全局别名映射到 count/pixels/IoU 等论文指标。
            metric_row = {}
            for item in rows:
                metric_name = item.get(metric_col)
                if metric_name:
                    metric_row[str(metric_name)] = item.get(value_col)
            return metric_row
        return rows[-1]

    def parse_particle_stats_array(self, array: np.ndarray) -> dict[str, float]:
        # 颗粒统计文件在不同阶段保存格式不完全一致：可能是 dict、结构化数组、二维 key-value 表，
        # 也可能是混合 object 数组（里面同时有 threshold_method 这类字符串字段和数值字段）。
        # 这里尽量按字段名解析；若字段名不可用，再只抽取可转成 float 的数值，避免字符串导致整张统计图失败。
        numpy = ensure_numpy()
        arr = np.asarray(array)
        if arr.dtype.names:
            row = arr.reshape(-1)[-1]
            return self.map_stats_fields({name: row[name] for name in arr.dtype.names})
        if arr.dtype == object:
            obj = arr.item() if arr.shape == () else arr.reshape(-1)[-1]
            if isinstance(obj, dict):
                mapped = self.map_stats_fields(obj)
                if mapped:
                    return mapped

            flat = arr.reshape(-1)
            # 兼容形如 [[key, value], ...] 的 object 表；这种结构里常会出现 threshold_method 字符串字段。
            if arr.ndim == 2 and arr.shape[1] >= 2:
                row = {}
                for item in arr:
                    key = item[0]
                    if isinstance(key, str):
                        row[key] = item[1]
                mapped = self.map_stats_fields(row)
                # 如果已经识别为 key-value 表，但字段别名仍未匹配，就返回空字典；
                # 不能继续按“前几个数字”兜底，否则 threshold/height/width 会被误当成 count/IoU。
                return mapped

            # 兼容形如 [key, value, key, value, ...] 的扁平 object 表。
            row = {}
            for idx in range(0, max(0, flat.size - 1), 2):
                key = flat[idx]
                if isinstance(key, str):
                    row[key] = flat[idx + 1]
            mapped = self.map_stats_fields(row)
            if row:
                # 扁平 key-value 表同样只接受字段名映射结果，避免字符串表误走数字兜底。
                return mapped

            numeric_values = []
            for value in flat:
                number = self.to_float(value)
                if number is not None:
                    numeric_values.append(number)
            numeric = numpy.asarray(numeric_values, dtype=numpy.float64)
        else:
            numeric = numpy.asarray(arr, dtype=numpy.float64).reshape(-1)
        keys = ("count", "pixels", "iou", "precision", "recall", "f1")
        return {key: float(numeric[idx]) for idx, key in enumerate(keys) if idx < numeric.size}

    def parse_particle_raw_stats_array(self, array: np.ndarray) -> dict:
        """从 npy 中提取原始 key-value 统计字段，避免 GT 字段被预测字段映射规则覆盖。"""

        arr = np.asarray(array)
        if arr.dtype.names:
            row = arr.reshape(-1)[-1]
            return {name: row[name] for name in arr.dtype.names}
        if arr.dtype == object:
            obj = arr.item() if arr.shape == () else arr.reshape(-1)[-1]
            if isinstance(obj, dict):
                return obj
            flat = arr.reshape(-1)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                row = {}
                for item in arr:
                    key = item[0]
                    if isinstance(key, str):
                        row[key] = item[1]
                return row
            row = {}
            for idx in range(0, max(0, flat.size - 1), 2):
                key = flat[idx]
                if isinstance(key, str):
                    row[key] = flat[idx + 1]
            return row
        return {}

    def map_stats_fields(self, row: dict) -> dict[str, float]:
        # 字段名统一做大小写和符号归一化，兼容 IoU / iou、F1 / f1_score 等不同保存习惯。
        normalized_keys = {normalize_name(key): key for key in row.keys()}
        mapped: dict[str, float] = {}
        for canonical, aliases in self.cfg.PARTICLE_STATS_FIELD_ALIASES.items():
            for alias in aliases:
                real_key = row.get(alias, None)
                if alias not in row and normalize_name(alias) in normalized_keys:
                    real_key = row[normalized_keys[normalize_name(alias)]]
                number = self.to_float(real_key)
                if number is not None:
                    mapped[canonical] = number
                    break
        return mapped

    def map_gt_particle_stats_fields(self, row: dict) -> dict[str, float]:
        """把原始统计字段映射成 GT 条形图指标；GT 自身二值重叠指标固定为 1。"""

        normalized_keys = {normalize_name(key): key for key in row.keys()}
        mapped: dict[str, float] = {}
        for canonical, aliases in self.cfg.PARTICLE_GT_STATS_FIELD_ALIASES.items():
            for alias in aliases:
                real_key = row.get(alias, None)
                if alias not in row and normalize_name(alias) in normalized_keys:
                    real_key = row[normalized_keys[normalize_name(alias)]]
                number = self.to_float(real_key)
                if number is not None:
                    mapped[canonical] = number
                    break
        self_metric = float(self.cfg.PARTICLE_GT_SELF_METRIC_VALUE)
        mapped.update({"iou": self_metric, "precision": self_metric, "recall": self_metric, "f1": self_metric})
        return mapped

    def to_float(self, value) -> float | None:
        """把 numpy 标量、单元素数组或字符串数值转成 float；非数值字段返回 None。"""

        if value is None:
            return None
        try:
            arr = ensure_numpy().asarray(value)
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    def to_float_loose(self, value) -> float | None:
        """更宽松地解析 CSV 数值，兼容千分位逗号和百分号，供 metrics_summary 最大值统计使用。"""

        number = self.to_float(value)
        if number is not None:
            return number
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        # metrics_summary.csv 中有些列可能写成 "1,234.5" 或 "98.7%"；
        # 这里仅用于比较大小，百分号按数值 98.7 比较，并保留原始文本写回输出表。
        cleaned = text.replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1].strip()
        try:
            return float(cleaned)
        except Exception:
            return None


__all__ = [
    "AllHandlePipeline",
    "GroupContext",
    "SampleBundle",
    "ensure_matplotlib",
    "ensure_numpy",
    "normalize_name",
    "safe_name",
]
