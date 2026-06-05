from pathlib import Path


class global_data:
    class all_handle:
        """
        all_handle 只负责“已完成实验结果”的统一后处理，不重新训练、不加载模型。
        这里集中保存所有可调参数：数据目录、八个对比实验的路径映射、图例文字、
        坐标轴 label、色条 label、配色、输出目录与样本筛选规则。
        """

        # =========================
        # 数据目录与输出目录
        # =========================
        # 原始实验结果根目录；pipeline.py 会在该目录下寻找八个对比实验的 predict_all/test_all 结果。
        # DATA_ROOT_DIR = Path(
        #     r"D:\BaiduSyncdisk\AYanJiuSheng\data\train_datas\root\autodl-tmp\train_datas"
        # )
        DATA_ROOT_DIR = Path(
                r"/study_datas/train_datas/root/autodl-tmp/train_datas/"
        )
        # 汇总图默认保存到 all_handle/output，避免覆盖各个原始实验目录中的结果图。
        DEFAULT_OUTPUT_ROOT_DIR = Path(__file__).resolve().parent / "output"
        # 输出根目录覆盖项：保持 None 时仍使用 DEFAULT_OUTPUT_ROOT_DIR；
        # 如果希望把所有结果写到其它位置，就把这里改成 Path(r"你的输出目录") 或字符串路径。
        OUTPUT_ROOT_DIR =Path(r"/study_datas/train_all_datas/")

        # 输出目录中的子文件夹名；若后续想把汇总结果按论文图号重排，只需要改这里。
        ENERGY_OUTPUT_DIR_NAME = "01_energy_spectrum"
        ERROR_MAP_OUTPUT_DIR_NAME = "02_error_maps"
        HIST_OUTPUT_DIR_NAME = "03_error_histograms"
        COMPOSITE_OUTPUT_DIR_NAME = "04_composite_panels"
        SUMMARY_OUTPUT_DIR_NAME = "00_summary"
        # 对比指标汇总表输出目录。CSV 没有 sheet 概念，所以 pipeline.py 会同时输出：
        # 1）一个扁平总表 CSV；2）每个类别一个 CSV；3）一个真正带类别 sheet 的 xlsx 工作簿。
        METRIC_TABLE_OUTPUT_DIR_NAME = "05_metric_tables"
        METRIC_TABLE_FLAT_CSV_NAME = "comparison_metrics.csv"
        METRIC_TABLE_SHEET_CSV_DIR_NAME = "category_csv_sheets"
        METRIC_TABLE_WORKBOOK_NAME = "comparison_metrics.xlsx"
        # metrics_summary.csv 里包含每个类别/样本的多行统计；这里单独输出一套对比表，
        # 避免和 ALL_CLASS_flow / ALL_CLASS_IMAGE_PAIR 汇总表混在一起。
        METRIC_SUMMARY_FILE_NAMES = ("metrics_summary.csv", "metrics_summary.CSV")
        METRIC_SUMMARY_FLAT_CSV_NAME = "metrics_summary_comparison.csv"
        METRIC_SUMMARY_SHEET_CSV_DIR_NAME = "metrics_summary_category_csv_sheets"
        METRIC_SUMMARY_WORKBOOK_NAME = "metrics_summary_comparison.xlsx"
        # metrics_summary.csv 的前 11 列按用户要求直接取一行的值；第 12 列开始按每列最大值汇总。
        METRIC_SUMMARY_FIXED_COLUMN_COUNT = 11
        # 输出表开头补充这些定位列，便于在同一个 CSV/xlsx 中比较不同对比组、class、split、类别和实验。
        METRIC_SUMMARY_METADATA_COLUMNS = ("comparison", "class", "split", "category", "experiment")

        # 控制本次要生成哪些输出阶段：
        # - None 或 "all"：生成 01_energy_spectrum、02_error_maps、03_error_histograms、04_composite_panels 全部；
        #   05_metric_tables 不在 group 步骤里重复运行，而是在所有 group 处理完成后统一写出。
        # - 字符串：例如 "02_error_maps"；
        # - 元组/列表：例如 ("01_energy_spectrum", "03_error_histograms")；
        #   也可以把独立重跑阶段组合进去，例如 ("tbl_profile_overlay", "particle_stats_metrics", "flow_u_epe_hist_overlay")。
        # 注意：这些独立重跑阶段在 None/all 默认模式下不会额外执行，因为它们已经分别包含在标准阶段中：
        # tbl_profile_overlay 属于 04_composite_panels，particle_stats_metrics 属于 04_composite_panels，
        # flow_u_epe_hist_overlay 属于 03_error_histograms，tbl_02_error_map 属于 02_error_maps。
        # 支持目录名和短名混用：energy_spectrum / error_maps / error_histograms / composite_panels。
        # 如果只想直接处理 TBL 剖面图，设置 OUTPUT_STAGE_FILTER = "tbl_profile_overlay"；
        # 如果只想重跑 TBL 的 v 方向且去掉 bicubic-hs 的剖面图，
        # 设置 OUTPUT_STAGE_FILTER = "tbl_v_profile_overlay_without_bicubic_hs"；
        # 如果只想重跑 TBL/TWCF 的能量谱图，设置 OUTPUT_STAGE_FILTER = "tbl_twcf_energy_spectrum"；
        # 如果只想处理颗粒统计条形图，设置 OUTPUT_STAGE_FILTER = "particle_stats_metrics"；
        # 如果只想处理 *_flow_u_epe_hist_overlay.png，设置 OUTPUT_STAGE_FILTER = "flow_u_epe_hist_overlay"。
        # 如果只想处理 TBL 的 02_error_maps，设置 OUTPUT_STAGE_FILTER = "tbl_02_error_map"。
        # OUTPUT_STAGE_FILTER =None
        OUTPUT_STAGE_FILTER =("tbl_v_profile_overlay_without_bicubic_hs")
        OUTPUT_STAGE_ALIASES = {
            "01_energy_spectrum": "energy_spectrum",
            "energy_spectrum": "energy_spectrum",
            "energy": "energy_spectrum",
            "02_error_maps": "error_maps",
            "error_maps": "error_maps",
            "maps": "error_maps",
            "03_error_histograms": "error_histograms",
            "error_histograms": "error_histograms",
            "histograms": "error_histograms",
            "hist": "error_histograms",
            "04_composite_panels": "composite_panels",
            "composite_panels": "composite_panels",
            "composites": "composite_panels",
            "panels": "composite_panels",
            "05_metric_tables": "metric_tables",
            "metric_tables": "metric_tables",
            "metrics": "metric_tables",
            "tables": "metric_tables",
            "tbl_profile_overlay": "tbl_profile_overlay",
            "profile_overlay": "tbl_profile_overlay",
            "tbl_profile": "tbl_profile_overlay",
            "profile": "tbl_profile_overlay",
            "tbl_v_profile_overlay_without_bicubic_hs": "tbl_v_profile_overlay_without_bicubic_hs",
            "tbl_v_profile_without_bicubic_hs": "tbl_v_profile_overlay_without_bicubic_hs",
            "tbl_v_without_bicubic_hs": "tbl_v_profile_overlay_without_bicubic_hs",
            "tbl_twcf_energy_spectrum": "tbl_twcf_energy_spectrum",
            "tbl_twcf_energy": "tbl_twcf_energy_spectrum",
            "tbl_energy_spectrum": "tbl_twcf_energy_spectrum",
            "twcf_energy_spectrum": "tbl_twcf_energy_spectrum",
            "particle_stats_metrics": "particle_stats_metrics",
            "particle_metrics": "particle_stats_metrics",
            "particle_bar": "particle_stats_metrics",
            "particle_bars": "particle_stats_metrics",
            "stats_metrics": "particle_stats_metrics",
            "particle_binary_count": "particle_binary_count",
            "particle_binary_counts": "particle_binary_count",
            "binary_count": "particle_binary_count",
            "binary_counts": "particle_binary_count",
            "flow_u_epe_hist_overlay": "flow_u_epe_hist_overlay",
            "flow_u_epe": "flow_u_epe_hist_overlay",
            "u_epe_hist": "flow_u_epe_hist_overlay",
            "epe_hist_overlay": "flow_u_epe_hist_overlay",
            "tbl_02_error_map": "tbl_02_error_map",
            "tbl_02_error_maps": "tbl_02_error_map",
            "tbl_error_map": "tbl_02_error_map",
            "tbl_error_maps": "tbl_02_error_map",
            "all": "all",
        }

        # =========================
        # 数据层级与运行范围
        # =========================
        # 兼容 class_1/class_2 与用户描述里的 class1/class2；扫描时会做大小写和下划线归一化。
        CLASS_NAMES = ("class_1", "class_2")
        SPLIT_NAMES = ("predict_all", "test_all")
        TRAIN_MODE_DIR_NAMES = ("mixed_all_classes", "problem_class2_raft_piv")
        RAFT_DIR_NAME = "RAFT"
        # 默认倍率目录；单个实验可用 EXPERIMENT_SCALE_DIR_NAMES 覆盖。
        SCALE_DIR_NAME = "scale_4"

        # None 表示不限制类别；例如只想跑 backstep/cylinder 时可改为 ("backstep", "cylinder")。
        CATEGORY_FILTER = None
        # None 表示不限制样本；生成大拼图时如果只想快速看一两个样本，可改成 ("sample_0000",) 或具体样本名。
        SAMPLE_FILTER = None
        # 组合图可能很多，None 表示每个类别所有共同样本都生成；调试时可改为 1 或 3。
        MAX_SAMPLE_COMPOSITES_PER_CATEGORY = None
        # 是否保存后处理过程中的 npy 汇总文件。按用户最新要求默认关闭；
        # 关闭后 pipeline 仍会读取原始实验目录里的 npy 数据，但不会在输出目录额外保存 .npy 文件。
        SAVE_NPY_OUTPUTS = False
        # 续跑起点配置：None 表示从头跑；例如当前停在 group 30/46 step 3/5: error_maps，
        # 就设置 RESUME_GROUP_INDEX = 30，RESUME_STEP_NAME = "error_maps"。
        # RESUME_STEP_NAME 优先级高于 RESUME_STEP_INDEX；只对 RESUME_GROUP_INDEX 对应的 group 生效，
        # 后续 group 会从第一个 step 正常继续跑。
        RESUME_GROUP_INDEX = None
        RESUME_STEP_INDEX = None
        RESUME_STEP_NAME = None
        # 命令行进度显示开关：全量跑 class/split/category 时耗时较长，开启后会打印当前处理到第几个 group。
        PROGRESS_ENABLED = True
        # 是否显示 group 内部的大步骤进度，例如 energy_spectrum / histograms / error_maps / composites。
        PROGRESS_SHOW_STEPS = True
        # 是否在进度日志里显示每个 step、每个 group 和总任务的运行时间。
        PROGRESS_SHOW_RUNTIME = True
        # 运行时间保留的小数位；只影响日志和 summary，不影响任何数值结果。
        PROGRESS_RUNTIME_DECIMALS = 2
        # 进度输出前缀统一放到全局变量，方便后续改成自己的日志格式。
        PROGRESS_PREFIX = "[all_handle]"


        # =========================
        # 八个对比实验与图例
        # =========================
        # key 是内部稳定标识；label 是最终出现在图例上的英文文本，按用户要求统一放在全局变量中。
        # 现在把 PIV_A_Esrgan_v4 明确标成 ESRuRAFT-PIV x4，并新增 PIV_A_Esrgan_v_SCALE_8 作为 ESRuRAFT-PIV x8。
        EXPERIMENT_KEYS = (
            "bicubic_widim",
            "bicubic_hs",
            "bicubic_raft",
            "bicubic_searaft",
            "srgan_raft",
            "swinir_raft",
            "esrgan_raft",
            "PIV_A_Esrgan_v4",
            "PIV_A_Esrgan_v_SCALE_8",
        )
        # 图例显示顺序单独控制：绘图时可以为了遮挡关系调整绘制顺序，
        # 但 legend 必须始终按这里从上到下排列。
        LEGEND_EXPERIMENT_ORDER = (
            "bicubic_widim",
            "bicubic_hs",
            "bicubic_raft",
            "bicubic_searaft",
            "srgan_raft",
            "swinir_raft",
            "esrgan_raft",
            "PIV_A_Esrgan_v4",
            "PIV_A_Esrgan_v_SCALE_8",
        )
        EXPERIMENT_LABELS = {
            "bicubic_widim": "bicubic-widim",
            "bicubic_hs": "bicubic-hs",
            "bicubic_raft": "bicubic-raft",
            "bicubic_searaft": "bicubic-searaft",
            "srgan_raft": "srgan-raft",
            "swinir_raft": "swinir-raft",
            "esrgan_raft": "esrgan-raft",
            "PIV_A_Esrgan_v4": "ESRuRAFT-PIV x4",
            "PIV_A_Esrgan_v_SCALE_8": "ESRuRAFT-PIV x8",
        }
        # 两套图分开生成：
        # 1）eight_experiments：原始八组对比实验，PIV_A_Esrgan_v4 在这套图里显示为 ESRuRAFT-PIV；
        # 2）eight_experiments_without_widim_hs：去掉 bicubic-widim 和 bicubic-hs 的补充对比图；
        # 3）scale_x4_x8：只比较 ESRuRAFT-PIV x4 与 ESRuRAFT-PIV x8。
        COMPARISON_GROUPS = {
            "eight_experiments": (
                "bicubic_widim",
                "bicubic_hs",
                "bicubic_raft",
                "bicubic_searaft",
                "srgan_raft",
                "swinir_raft",
                "esrgan_raft",
                "PIV_A_Esrgan_v4",
            ),
            "eight_experiments_without_widim_hs": (
                "bicubic_raft",
                "bicubic_searaft",
                "srgan_raft",
                "swinir_raft",
                "esrgan_raft",
                "PIV_A_Esrgan_v4",
            ),
            "scale_x4_x8": (
                "PIV_A_Esrgan_v4",
                "PIV_A_Esrgan_v_SCALE_8",
            ),
        }
        # 每套对比图的图例顺序；legend 从上到下严格按这里排列。
        COMPARISON_GROUP_LEGEND_ORDER = {
            "eight_experiments": COMPARISON_GROUPS["eight_experiments"],
            "eight_experiments_without_widim_hs": COMPARISON_GROUPS["eight_experiments_without_widim_hs"],
            "scale_x4_x8": COMPARISON_GROUPS["scale_x4_x8"],
        }
        # 同一个实验在不同对比组里可以显示不同 label：x4 在八组对比里是 ESRuRAFT-PIV，在倍率对比里是 ESRuRAFT-PIV x4。
        COMPARISON_GROUP_LABELS = {
            "eight_experiments": {
                "PIV_A_Esrgan_v4": "ESRuRAFT-PIV",
            },
            "eight_experiments_without_widim_hs": {
                "PIV_A_Esrgan_v4": "ESRuRAFT-PIV",
            },
            "scale_x4_x8": {
                "PIV_A_Esrgan_v4": "ESRuRAFT-PIV x4",
                "PIV_A_Esrgan_v_SCALE_8": "ESRuRAFT-PIV x8",
            },
        }
        # 最小实验数量限制；倍率对比必须同时找到 x4 与 x8，避免只画单独 x4 的无意义图。
        COMPARISON_GROUP_MIN_EXPERIMENTS = {
            "eight_experiments": 1,
            "eight_experiments_without_widim_hs": 1,
            "scale_x4_x8": 2,
        }
        # 实际磁盘目录名与用户口径不完全一致，因此这里显式建立映射；若目录改名，只改这一个位置。
        EXPERIMENT_DIR_NAMES = {
            "bicubic_widim": "ESRuRAFT_PIV_Groundv_bicubic_widim",
            "bicubic_hs": "ESRuRAFT_PIV_Groundv_bicubic_hs",
            "bicubic_raft": "ESRuRAFT_PIV_Groundv_bicubic_raft",
            "bicubic_searaft": "ESRuRAFT_PIV_Groundv_bicubic_searaft",
            "srgan_raft": "ESRuRAFT_PIV_Groundv_srgan_raft",
            "swinir_raft": "ESRuRAFT_PIV_Groundv_swinir_raft",
            "esrgan_raft": "ESRuRAFT_PIV_Groundv_esrgan_raft",
            "PIV_A_Esrgan_v4": "PIV_A_Esrgan_v4",
            "PIV_A_Esrgan_v_SCALE_8": "PIV_A_Esrgan_v_SCALE_8",
        }
        # 新增实验有时会直接用用户口径的短目录名保存；这里保留备用目录名。
        # pipeline.py 会优先使用 EXPERIMENT_DIR_NAMES，找不到时再尝试这些别名，避免因为目录命名差异漏画。
        EXPERIMENT_DIR_NAME_ALIASES = {
            "bicubic_widim": ("bicubic_widim",),
            "bicubic_hs": ("bicubic_hs",),
            "bicubic_raft": ("bicubic_raft",),
            "bicubic_searaft": ("bicubic_searaft",),
            "srgan_raft": ("srgan_raft",),
            "swinir_raft": ("swinir_raft",),
            "esrgan_raft": ("esrgan_raft",),
        }
        # 大多数实验在 scale_4 目录；x8 消融结果单独保存在 scale_8，所以对每个实验做独立倍率映射。
        EXPERIMENT_SCALE_DIR_NAMES = {
            "bicubic_widim": "scale_4",
            "bicubic_hs": "scale_4",
            "bicubic_raft": "scale_4",
            "bicubic_searaft": "scale_4",
            "srgan_raft": "scale_4",
            "swinir_raft": "scale_4",
            "esrgan_raft": "scale_4",
            "PIV_A_Esrgan_v4": "scale_4",
            "PIV_A_Esrgan_v_SCALE_8": "scale_8",
        }
        # 论文常用、色盲友好的配色；每个实验固定一种颜色，所有图例保持一致。
        # 按用户要求：ESRuRAFT-PIV（PIV_A_Esrgan_v4）统一使用红色；
        # bicubic-hs 从原来的橙红色改为紫色，避免和 ESRuRAFT-PIV 的红色混淆。
        EXPERIMENT_COLORS = {
            "bicubic_widim": "#0072B2",
            "bicubic_hs": "#785EF0",
            "bicubic_raft": "#009E73",
            "bicubic_searaft": "#56B4E9",
            "srgan_raft": "#CC79A7",
            "swinir_raft": "#D55E00",
            "esrgan_raft": "#E69F00",
            "PIV_A_Esrgan_v4": "#D62728",
            "PIV_A_Esrgan_v_SCALE_8": "#000000",
        }
        # 误差直方图专用调色板：按用户给出的图 1 使用 Matplotlib 默认 tab10 风格颜色。
        # 该配色只影响误差直方图，普通图例仍使用 EXPERIMENT_COLORS。
        EXPERIMENT_HIST_COLORS = {
            "bicubic_widim": "#1f77b4",
            "bicubic_hs": "#ff7f0e",
            "bicubic_raft": "#2ca02c",
            "bicubic_searaft": "#17becf",
            "srgan_raft": "#8c564b",
            "swinir_raft": "#bcbd22",
            "esrgan_raft": "#9467bd",
            "PIV_A_Esrgan_v4": "#d62728",
            "PIV_A_Esrgan_v_SCALE_8": "#7f7f7f",
        }


        # =========================
        # 全局图形样式
        # =========================
        FIG_DPI = 300
        FIG_FORMATS = ("png", "svg")
        # 论文图默认优先使用 Times New Roman；如果当前系统/服务器没有安装该字体，
        # pipeline.py 会按 FONT_FAMILY_FALLBACKS 自动选择可用衬线字体，避免 Matplotlib 反复输出
        # "Font family 'Times New Roman' not found." 警告。
        FONT_FAMILY = "Times New Roman"
        FONT_FAMILY_FALLBACKS = ("DejaVu Serif", "Liberation Serif", "Nimbus Roman", "serif")
        AXIS_LABEL_SIZE = 12
        TICK_LABEL_SIZE = 10
        LEGEND_FONT_SIZE = 10
        PANEL_LABEL_SIZE = 12
        COLORBAR_LABEL_SIZE = 10
        # 所有色条 tick 统一格式：正数和 0 前补空白，让 " 1.0" 与 "-1.0" 这类正负范围对齐。
        COLORBAR_PAD_POSITIVE_TICKS = True
        COLORBAR_TICK_DECIMALS = 1
        # 所有 auto 色条默认用分位数范围，避免极少数尖峰像素把整张图的色条撑得过大。
        # center_zero=True 的误差图会使用 abs(value) 的该分位数，并保持 [-p, p] 对称；
        # 普通值图会使用双侧分位数 [0.25%, 99.75%]（当这里为 99.5 时）。
        # 如果需要恢复严格 min/max，把该值设为 None。
        COLORBAR_AUTO_PERCENTILE = 99.5
        # 误差直方图按参考图使用半透明填充色，不再给直方图和图例添加加粗边框。
        HIST_ALPHA = 0.55
        HIST_BINS = 201
        HIST_LINE_WIDTH = 0.0
        HIST_EDGE_DARKEN = 0.72
        HIST_EDGE_LINE_WIDTH = 0.0
        HIST_LEGEND_EDGE_LINE_WIDTH = 0.0
        HIST_DRAW_OUTLINE = False
        # 误差直方图的 npy 往往已经保存为“中心点 x + count”的直方图，
        # 不同实验如果各自使用不同 x 范围和 bin 宽度，直接叠加会导致同一张图里 count 峰值不可比。
        # 开启后，pipeline 会在绘图前把同一张图中的所有方法重分箱到统一公共 bins；
        # 原始 npy 不会被改动，只统一输出图和诊断 CSV 的统计口径。
        HIST_USE_SHARED_BINS = True
        # 公共 bins 的来源。widest_series 表示直接选取“所有对比方法中 x range 最大”的那组直方图作为模板，
        # 其它方法都重分箱到它的 x centers / bin width / range 上；这和误差图色条取最大范围的思路一致。
        HIST_SHARED_BIN_TEMPLATE_MODE = "widest_series"
        # 公共 bin 宽度默认取当前图中“最粗”的原始 bin 宽，例如 backstep 中 bicubic-hs 的
        # x range=[-7.34, 7.34]、bin width≈0.0734 会成为其它方法共同使用的 bin 宽。
        # 仅当 HIST_SHARED_BIN_TEMPLATE_MODE 不是 widest_series 时才会使用该模式。
        HIST_SHARED_BIN_WIDTH_MODE = "coarsest"
        # 以 0 为中心的误差直方图使用对称 x 范围；EPE 是非负量，不做 0 对称处理。
        HIST_SHARED_BIN_SYMMETRIC_AXIS_KINDS = ("flow", "flow_u", "particle", "vorticity")
        # 统一 bins 后，公共模板两端可能有大量 0 count 的空白 bin，导致图中主峰被挤得很小。
        # 开启后，绘图的 x 轴显示范围只根据实际有 count 的 bin 自动缩小；
        # flow/particle/vorticity/Delta u 按 0 对称，EPE 只从 0 到最大值。
        HIST_USE_DYNAMIC_X_LIMITS = True
        HIST_DYNAMIC_X_LIMIT_MIN_COUNT = 0.0
        HIST_DYNAMIC_X_LIMIT_MARGIN_RATIO = 0.0
        # 新增“2D + 局部 3D”合图：2D 子图仍显示完整/当前二维范围，
        # 3D 子图只显示直方图 count 变化明显的 x 区域，避免长尾把 3D 主峰挤得太窄。
        # 明显区域按“所有实验 count 叠加后超过峰值比例”的 bin 自动确定；
        # 例如峰值附近为主要变化区时，会自动得到类似 [-5, 5] 这样的局部范围。
        WATERFALL_3D_FOCUS_COMPOSITE_ENABLED = True
        HIST_FOCUS_X_LIMIT_THRESHOLD_RATIO = 0.005
        HIST_FOCUS_X_LIMIT_MARGIN_RATIO = 0.08
        # 为避免噪声造成过窄范围，局部 3D 至少保留当前动态显示范围的一定比例。
        HIST_FOCUS_X_LIMIT_MIN_SPAN_RATIO = 0.18
        # 误差直方图里 ESRuRAFT-PIV 必须最后绘制并处在最高层，避免被其它半透明柱子覆盖；
        # 倍率对比图里的 x4/x8 也属于本文方法，因此同样放到顶层。
        HIST_TOP_EXPERIMENT_KEYS = ("PIV_A_Esrgan_v4", "PIV_A_Esrgan_v_SCALE_8")
        # 顶层实验不仅最后绘制、最高 zorder，还单独降低“透明感”（提高 alpha），
        # 这样 ESRuRAFT-PIV 会位于所有误差直方图最上层，并且比其它半透明柱子更突出。
        # 若后续想恢复所有方法同样透明度，把 HIST_HIGHLIGHT_EXPERIMENT_KEYS 改成 () 即可。
        HIST_HIGHLIGHT_EXPERIMENT_KEYS = ("PIV_A_Esrgan_v4", "PIV_A_Esrgan_v_SCALE_8")
        HIST_HIGHLIGHT_ALPHA = 0.78
        # flow_u_epe_hist_overlay 是左右两张子图，右图 y 轴 Count 容易贴到左图；
        # 这里单独控制两个子图之间的横向间隔。
        FLOW_U_EPE_HIST_WSPACE = 0.32
        # 八组实验的 flow_u_epe_hist_overlay 图例如果放在右侧 EPE 子图内部，会和 0 附近的高峰挤在一起；
        # 因此当实验数量达到阈值时，把图例放到右侧子图外部。下面几个参数可单独调图宽、锚点和图例列数。
        # 普通 *_flow_u_epe_hist_overlay.png 的图例放回右侧 EPE 子图内部；
        # 如果放到图外，横向空间会被 legend 拉宽，论文排版时右侧空白过多。
        FLOW_U_EPE_HIST_LEGEND_OUTSIDE = False
        FLOW_U_EPE_HIST_LEGEND_OUTSIDE_MIN_SERIES = 7
        FLOW_U_EPE_HIST_OUTSIDE_LEGEND_FIG_WIDTH = 10.2
        FLOW_U_EPE_HIST_LEGEND_OUTSIDE_LOC = "upper left"
        FLOW_U_EPE_HIST_LEGEND_OUTSIDE_BBOX = (1.02, 1.0)
        FLOW_U_EPE_HIST_LEGEND_OUTSIDE_NCOL = 1
        # 3D 瀑布图开关与统一风格；只控制新增的 *_waterfall_3d.png/svg，
        # 不改变原来的二维能谱图和误差直方图。
        WATERFALL_3D_ENABLED = True
        # 单独 3D 瀑布图底部需要足够空间显示带单位的 x-label，例如 "Flow error [px]"。
        WATERFALL_3D_FIG_SIZE = (9.2, 7.2)
        # 单独 3D 图把坐标轴整体往上挪一点，给底部 x-label 的 "[px]" 留出空间；
        # 这个参数只作用于 *_waterfall_3d 单图，不影响 2D+3D 合图。
        WATERFALL_3D_SUBPLOT_BOTTOM = 0.16
        WATERFALL_3D_PAIR_FIG_SIZE = (10.8, 4.6)
        WATERFALL_3D_ELEV = 24
        # 3D 视角默认把 x 轴放到左前侧，避免 y 轴方向的实验名抢占主要视觉位置。
        WATERFALL_3D_AZIM = -60
        # 3D 瀑布图使用正交投影，减少透视带来的 z 轴倾斜观感，让 z 轴更接近竖直 90 度。
        # Matplotlib 支持 "ortho" 和 "persp"；如果想恢复透视效果可改回 "persp"。
        WATERFALL_3D_PROJECTION = "ortho"
        # 3D 坐标盒比例：适当放大 x/z 方向，避免合图中 3D 曲线显得比左侧 2D 图小一圈。
        WATERFALL_3D_BOX_ASPECT = (1.28, 0.92, 0.92)
        # 3D 坐标盒在子图中的缩放；增大后合图右侧 3D 图会更接近左侧 2D 图的视觉大小。
        WATERFALL_3D_BOX_ZOOM = 1.16
        # Matplotlib 3D 默认会把 z 轴放在右侧；这里用私有轴布局参数把 z 轴挪到左侧，
        # 让图形更接近用户给出的瀑布图示例。若当前 Matplotlib 版本不支持，会自动忽略。
        WATERFALL_3D_FORCE_Z_AXIS_LEFT = True
        WATERFALL_3D_LINE_WIDTH = 1.45
        WATERFALL_3D_GRID_ALPHA = 0.18
        WATERFALL_3D_MAX_POINTS = 700
        # 3D 轴的 x-label 在倾斜视角下很容易落到画布外；这里把 labelpad 单独做成全局参数，
        # 既可以把 "Delta u [px]" / "EPE [px]" 往坐标轴内收，也不影响二维图的 label 间距。
        WATERFALL_3D_X_LABEL_PAD = 2
        WATERFALL_3D_LEGEND_FONT_SIZE = 7
        WATERFALL_3D_LEGEND_LOC = "upper left"
        # 单独 3D 瀑布图的图例位置；第二个值控制上下，第一个值越小图例越靠近图体。
        # 用户反馈图例离图太远，因此这里从旧版 1.42 收到 1.16。
        WATERFALL_3D_LEGEND_BBOX = (0.92, 1.0)
        WATERFALL_3D_LEGEND_LABEL_SPACING = 0.35
        WATERFALL_3D_LEGEND_HANDLE_LENGTH = 2.0
        # 2D+3D “大图”尺寸与子图间距：左侧复用原二维图，右侧放新增 3D 瀑布图。
        WATERFALL_3D_COMPOSITE_FIG_SIZE = (12.8, 4.6)
        # 合图右侧 3D 子图给更多宽度，使其视觉面积接近左侧 2D 图，不再显得偏小。
        WATERFALL_3D_COMPOSITE_WIDTH_RATIOS = (1.0, 1.36)
        # 右侧 3D 图的 z 轴 label 会放在 3D 子图左侧；合图中适当加大 wspace，
        # 避免 Count / log10(...) 这类 z-label 与左侧二维图或 z 轴刻度挤在一起。
        # 合图左右两张图的间距只控制 2D/3D 子图距离，不再用它解决 z-label 重叠问题；
        # z-label 位置由 WATERFALL_3D_COMPOSITE_LEFT_Z_LABEL_X 单独控制。
        WATERFALL_3D_COMPOSITE_WSPACE = 0.18
        # flow_u_epe_hist_overlay 的大图比较特殊：第一行是原来的左右二维图，
        # 第二行是对应的两个 3D 瀑布图，因此单独给更高的画布和行列间距。
        # flow_u_epe 的 2D+3D 合图是两行布局：第一行二维图，第二行 3D 图。
        # 第二行如果和第一行等高，3D 盒体会显得明显偏小，x-label 也容易被挤到画布外；
        # 因此这里单独加高画布，并让第二行拿到更多高度。
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_FIG_SIZE = (12.8, 10.8)
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_WSPACE = 0.26
        # 控制第一行 2D 与第二行 3D 的上下空白；0.08 在紧凑排布和 3D x-label 显示空间之间较均衡。
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_HSPACE = 0.08
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_HEIGHT_RATIOS = (1.0, 1.55)
        # flow_u_epe 的 2D+3D 合图需要更大的底部边距，否则第二行 3D 图旋转后的 x-label 会被裁掉。
        # right 留到 0.98 是因为这张合图的图例改为放回右上二维子图内部，不再需要额外的图外 legend 空间。
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_LEFT = 0.06
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_RIGHT = 0.98
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_TOP = 0.98
        WATERFALL_3D_FLOW_U_EPE_COMPOSITE_BOTTOM = 0.12
        # 普通 flow_u_epe_hist_overlay 仍可使用图外 legend；但 2D+3D 合图已有两行，
        # 外置 legend 会显得像跑到画布外，所以这里单独控制合图版本把图例放回图内。
        FLOW_U_EPE_HIST_COMPOSITE_LEGEND_OUTSIDE = False
        # flow_u_epe 的 2D+3D 合图第二行是 3D 坐标轴，z 轴 Count 用 text2D 手动绘制。
        # 这里单独控制第二行 3D Count 的横向位置，让它和第一行二维图的 Count label 上下对齐；
        # 只影响 *_flow_u_epe_hist_overlay_2d_3d_composite，不影响其它 3D 瀑布图。
        WATERFALL_3D_FLOW_U_EPE_LEFT_Z_LABEL_X = -0.10
        # 下排 3D Count label 改用 figure 坐标与上排 2D Count label 对齐；
        # 这个偏移量用于最后细调，0 表示完全使用上排 2D y-label 的横坐标。
        WATERFALL_3D_FLOW_U_EPE_Z_LABEL_FIG_X_OFFSET = 0.0
        # y 方向已经由图例说明实验顺序，因此默认隐藏 y 轴刻度文字和 y 轴 label，避免和图例重复且挤在一起。
        WATERFALL_3D_SHOW_Y_TICK_LABELS = False
        WATERFALL_3D_SHOW_Y_LABEL = False
        WATERFALL_3D_HIDE_Y_AXIS_LINE = True
        WATERFALL_3D_USE_LEFT_Z_LABEL_TEXT = True
        # z-label 使用 2D 文本固定在 3D 坐标轴左侧；负值表示继续向左移，
        # 用来避开 z 轴刻度数字，解决 Count 和 300000/400000 等刻度挤在一起的问题。
        WATERFALL_3D_LEFT_Z_LABEL_X = -0.16
        # 2D+3D 合图里右侧 3D 子图更窄，z-label 更容易和 z 轴刻度挤在一起；
        # 因此合图单独使用更靠左的位置，而不增加两张图之间的间隔。
        WATERFALL_3D_COMPOSITE_LEFT_Z_LABEL_X = -0.24
        WATERFALL_3D_LEFT_Z_LABEL_Y = 0.55
        WATERFALL_3D_Z_TICK_PAD = 2
        # 能谱跨度很大，默认把 x/z 都转成 log10 后画 3D 瀑布图，这样不同曲线不会被数量级压扁。
        ENERGY_WATERFALL_USE_LOG10 = True
        # 误差直方图默认保留原始 Count 高度；若后续峰值过高，可改成 True 使用 log10(Count + 1)。
        HIST_WATERFALL_USE_LOG10_COUNT = False
        IMAGE_CMAP = "viridis"
        # 光流/颗粒误差图使用 bwr 发散色图；在 pipeline.py 中会强制 vmin/vmax 关于 0 对称，
        # 因而 0 一定处于色条正中间，并且对应纯白色。
        ERROR_CMAP = "bwr"
        BINARY_CMAP = "gray"
        LINE_CHART_MARKER = "o"

        # =========================
        # 所有图的英文 label
        # =========================
        # 按用户要求：所有可修改的 label 都放在全局变量里，图上不出现中文字符。
        LEGEND_LABEL = "Experiment"
        GT_ENERGY_LABEL = "GT"
        # 能谱图中的 GT 按用户要求使用纯黑色，而不是灰色。
        GT_ENERGY_COLOR = "#000000"
        GT_ENERGY_LINESTYLE = "--"
        ENERGY_X_LABEL = "Wavenumber"
        ENERGY_Y_LABEL = "Energy spectrum"
        ENERGY_WATERFALL_X_LABEL = "log10(Wavenumber)"
        ENERGY_WATERFALL_Z_LABEL = "log10(Energy spectrum)"
        WATERFALL_3D_Y_LABEL = "Experiment"
        HIST_WATERFALL_Z_LABEL = "Count"
        HIST_WATERFALL_LOG_Z_LABEL = "log10(Count + 1)"
        # 能谱图图例使用带透明度的灰色底框，避免多条 log-log 曲线穿过图例文字后影响阅读。
        ENERGY_LEGEND_FRAME = True
        ENERGY_LEGEND_FACE_COLOR = "#E6E6E6"
        ENERGY_LEGEND_EDGE_COLOR = "#808080"
        ENERGY_LEGEND_ALPHA = 0.58
        ENERGY_LEGEND_FONT_SIZE = 8
        FLOW_VALUE_COLORBAR_LABEL = "Displacement [px]"
        FLOW_ERROR_COLORBAR_LABEL = "Error [px]"
        PARTICLE_VALUE_COLORBAR_LABEL = "Intensity"
        PARTICLE_ERROR_COLORBAR_LABEL = "Error"
        VORTICITY_VALUE_COLORBAR_LABEL = "Vorticity"
        VORTICITY_ERROR_COLORBAR_LABEL = "Vorticity error"
        HIST_Y_LABEL = "Count"
        FLOW_ERROR_HIST_X_LABEL = "Flow error [px]"
        FLOW_U_HIST_X_LABEL = "Delta u [px]"
        EPE_HIST_X_LABEL = "EPE [px]"
        PARTICLE_ERROR_HIST_X_LABEL = "Particle error"
        VORTICITY_ERROR_HIST_X_LABEL = "Vorticity error"
        PARTICLE_GRAY_HIST_X_LABEL = "Gray value"
        PARTICLE_GRAY_HIST_Y_LABEL = "Pixel count"
        PARTICLE_COUNT_X_LABEL = "Experiment"
        PARTICLE_COUNT_Y_LABEL = "Value"
        PARTICLE_METRIC_Y_LABEL = "Metric"

        # 面板文字同样集中管理，便于后续改成论文里的简写。
        LR_PANEL_LABEL = "LR"
        GT_PANEL_LABEL = "GT"
        SR_PANEL_LABEL = "SR"
        PREVIOUS_ROW_LABEL = "previous"
        NEXT_ROW_LABEL = "next"
        BLANK_PANEL_LABEL = ""
        FLOW_GT_PANEL_LABEL = "GT"
        VORTICITY_GT_PANEL_LABEL = "GT"
        FLOW_COMPONENT_LABELS = {
            "u": "u displacement",
            "v": "v displacement",
            "s": "speed",
        }
        FLOW_COMPONENT_ERROR_LABELS = {
            "u": "u error",
            "v": "v error",
            "s": "speed error",
        }
        # TBL 光流剖面图的所有可调文字和样式。
        # test_all 的 TBL 目录里已经保存了 u/v 剖面的 pred/gt npy，不需要重新计算光流；
        # pipeline.py 会把同一 sample 的所有对比实验叠加到一张剖面图里，便于横向比较。
        TBL_PROFILE_GT_LABEL = "GT"
        TBL_PROFILE_X_LABELS = {
            "u": "U displacement [px]",
            "v": "V displacement [px]",
        }
        TBL_PROFILE_Y_LABEL = "y [px]"
        TBL_PROFILE_REGION_LABELS = ("Laminar", "Transition", "Turbulent")
        # TBL 剖面图三条红色虚线的默认比例，与原始 evaluate 代码中的 TBL_PROFILE_COLUMN_RATIOS 保持一致。
        # 当 profile_columns.npy 的坐标系和顶部底图尺寸不一致时，pipeline.py 会用这个比例重新映射，
        # 避免把 Turbulent 位置错误拉到整张图最右边。
        TBL_PROFILE_COLUMN_RATIOS = (0.15, 0.265, 0.83)
        TBL_PROFILE_TOP_LABELS = {
            "u": "GT U",
            "v": "GT V",
        }
        # 顶部 GT 流场标题按论文剖面图样式集中配置；如果不想显示标题，可把对应值改成空字符串。
        TBL_PROFILE_TOP_TITLES = {
            "u": "Ground truth of horizontal direction",
            "v": "Ground truth of vertical direction",
        }
        # 顶部 GT 流场优先读取 profile_analysis 中已经裁到有效边界层高度的 npy，
        # 这样红色虚线、Laminar/Transition/Turbulent 标签和下方剖面完全使用同一个坐标系。
        TBL_PROFILE_TOP_FIELD_FILE_NAMES = {
            "u": "u_gt_profile_view.npy",
            "v": "v_gt_profile_view.npy",
        }
        # 额外保存一套不包含 bicubic-hs 的 TBL 剖面对比图，避免 bicubic-hs 噪声太大时压缩其它曲线差异。
        TBL_PROFILE_EXTRA_EXCLUDE_EXPERIMENT_KEYS = ("bicubic_hs",)
        TBL_PROFILE_EXTRA_SUFFIX = "_without_bicubic_hs"
        # 下面三个比例控制“上方 GT 流场 + 中间横向色条 + 下方三张剖面图”的高度分配。
        # 下方剖面图刻意放高一些，贴近用户给出的第三张参考图，曲线和坐标 label 更清楚。
        TBL_PROFILE_TOP_HEIGHT_RATIO = 1.05
        TBL_PROFILE_COLORBAR_HEIGHT_RATIO = 0.18
        TBL_PROFILE_CURVE_HEIGHT_RATIO = 3.20
        TBL_PROFILE_GT_COLOR = "#444444"
        TBL_PROFILE_GT_LINESTYLE = "--"
        TBL_PROFILE_GT_LINE_WIDTH = 1.4
        TBL_PROFILE_PRED_LINE_WIDTH = 1.25
        TBL_PROFILE_ALPHA = 0.95
        TBL_PROFILE_GRID_ALPHA = 0.25
        # 剖面图右侧图例文字较长（如 ESRuRAFT-PIV），适当加宽整张图和图例区域，避免文字越界。
        TBL_PROFILE_FIG_WIDTH_PER_REGION = 3.95
        TBL_PROFILE_FIG_HEIGHT = 9.8
        # TBL 剖面图顶部 GT 流场和中间色条/图例行之间不需要太大空白；
        # 调小 hspace 可以让第一行和第二行更贴近，同时保留下方三张剖面图的可读性。
        TBL_PROFILE_HSPACE = 0.26
        TBL_PROFILE_WSPACE = 0.18
        # TBL 剖面图色条 label 默认放在色条上方，避免 "Displacement [px]" 和下方剖面子图标题挨在一起。
        TBL_PROFILE_COLORBAR_LABEL_PAD = 3
        TBL_PROFILE_COLORBAR_LABEL_POSITION = "top"
        # TBL 剖面图中间行改成“左色条 + 右图例”，避免图例压住三张剖面曲线。
        # 左右宽度比例、间距和图例列数都放到全局变量里，后续可以按论文版面继续微调。
        TBL_PROFILE_COLORBAR_LEGEND_WIDTH_RATIOS = (1.05, 1.25)
        TBL_PROFILE_COLORBAR_LEGEND_WSPACE = 0.14
        TBL_PROFILE_LEGEND_LOC = "center"
        TBL_PROFILE_LEGEND_NCOL = 2
        TBL_PROFILE_X_MIN = None
        TBL_PROFILE_X_MAX = None
        TBL_PROFILE_Y_MIN = None
        TBL_PROFILE_Y_MAX = None
        # TBL v 方向位移幅值较小，普通自动范围会被个别方法拉得过宽，导致主体曲线挤在中间。
        # 因此按分量提供专属横坐标范围；这里把 v 方向固定为 [-2, 2]，
        # 会作用于 tbl_v_profile_overlay_without_bicubic_hs.png 等 v 方向剖面图。
        TBL_PROFILE_COMPONENT_X_LIMITS = {
            "v": (-4.0, 4.0),
        }
        PARTICLE_STAT_COUNT_LABEL = "count"
        PARTICLE_STAT_PIXEL_LABEL = "particle pixels"
        PARTICLE_STAT_MEAN_AREA_LABEL = "mean area"
        PARTICLE_STAT_IOU_LABEL = "IoU"
        PARTICLE_STAT_PRECISION_LABEL = "precision"
        PARTICLE_STAT_RECALL_LABEL = "recall"
        PARTICLE_STAT_F1_LABEL = "F1"
        THRESHOLD_LABEL = "T"
        # 颗粒统计组合图分成两张输出：
        # 1）particle_binary_stats_image_composite：只放 GT/SR 图和二值阈值图；
        # 2）particle_binary_stats_metrics_composite：单独放 GT 灰度直方图和统计条形图，画布更大，避免标签拥挤。
        PARTICLE_STATS_FIG_WIDTH_PER_COL = 5.70
        PARTICLE_STATS_IMAGE_FIG_HEIGHT = 10.0
        PARTICLE_STATS_METRIC_FIG_HEIGHT = 15.5
        # 颗粒图/阈值图横向排版单独使用紧凑参数，避免复用条形统计图的大画布后图像之间留白过多。
        # crop 图仍保持横向多列，但列宽、行距和 previous/next 的间隔都会明显收紧。
        PARTICLE_STATS_IMAGE_WIDTH_PER_COL = 1.35
        PARTICLE_STATS_IMAGE_COMPACT_FIG_HEIGHT = 4.90
        # 颗粒阈值图横向间距继续收紧，并和上下间距保持一致。
        PARTICLE_STATS_IMAGE_COMPACT_WSPACE = 0.015
        PARTICLE_STATS_IMAGE_COMPACT_HSPACE = 0.015
        PARTICLE_STATS_IMAGE_COMPACT_BLOCK_GAP_RATIO = 0.08
        PARTICLE_STATS_IMAGE_ROW_RATIO = 1.35
        PARTICLE_STATS_CHART_ROW_RATIO = 2.25
        PARTICLE_STATS_BLOCK_GAP_RATIO = 0.28
        PARTICLE_STATS_WSPACE = 0.58
        PARTICLE_STATS_HSPACE = 0.42
        PARTICLE_STATS_XTICK_ROTATION = 50
        PARTICLE_STATS_XTICK_LABEL_SIZE = 8
        # 颗粒统计条形图用不同颜色区分 GT 与各个实验，并通过图例说明；
        # 横轴不再显示长实验名，避免旋转标签挤占图像空间。
        PARTICLE_STATS_BAR_COLOR = "#4477AA"
        PARTICLE_STATS_GT_BAR_COLOR = "#666666"
        PARTICLE_STATS_LR_BAR_COLOR = "#999999"
        PARTICLE_STATS_EXPERIMENT_BAR_COLORS = {
            "__lr__": "#999999",
            "bicubic_widim": "#4477AA",
            "bicubic_hs": "#785EF0",
            "bicubic_raft": "#228833",
            "bicubic_searaft": "#56B4E9",
            "srgan_raft": "#CCBB44",
            "swinir_raft": "#D55E00",
            "esrgan_raft": "#66CCEE",
            "PIV_A_Esrgan_v4": "#D62728",
            "PIV_A_Esrgan_v_SCALE_8": "#111111",
        }
        PARTICLE_STATS_BAR_EDGE_COLOR = "#222222"
        PARTICLE_STATS_VALUE_LABEL_SIZE = 7
        PARTICLE_STATS_VALUE_FORMAT = "plain"
        PARTICLE_STATS_VALUE_DECIMALS = 4
        PARTICLE_STATS_SHOW_XTICK_LABELS = False
        PARTICLE_STATS_LEGEND_NCOL = 2
        PARTICLE_STATS_LEGEND_FONT_SIZE = 7.5
        # 颗粒统计条形图左侧 previous/next 行标签和 y 轴 label 都需要留出更大空间，
        # 避免二者挤在一起；数值越大，左侧留白越多。
        PARTICLE_STATS_ROW_LABEL_X = -0.23
        PARTICLE_STATS_SUBPLOTS_LEFT = 0.075
        PARTICLE_STATS_SUBPLOTS_RIGHT = 0.985
        # 颗粒统计条形图的图例放在每张子图内部；使用 "best" 让 Matplotlib 自动寻找空白区域。
        # 同时把 y 轴顶部留白比例调高，给图例和柱顶数值留出空间，避免压住柱子或挤到边框。
        # 条形统计图内部图例固定在右上角，避免 Matplotlib 的 "best" 自动把图例放到左上，
        # 与 count / particle pixels 面板 label 挤在一起。
        PARTICLE_STATS_LEGEND_LOC = "upper right"
        PARTICLE_STATS_LEGEND_BBOX = (0.985, 0.985)
        # y 轴顶部额外留白稍微加大，让右上角图例和柱顶数值都有空间。
        PARTICLE_STATS_Y_PAD_RATIO = 0.70
        PARTICLE_STATS_Y_PAD_MIN = 1.0
        # 灰度直方图中 T=... 文字相对阈值竖线的横向偏移，使用 x 轴坐标比例；
        # 默认略向右移动，避免文字贴着虚线。
        PARTICLE_GRAY_HIST_THRESHOLD_TEXT_DX = 0.015
        # 颗粒阈值化图的面板标签通常包含较长实验名，例如 bicubic-widim、ESRuRAFT-PIV；
        # 这些图本身列宽较窄，因此整张颗粒阈值对比图都使用更小字号，避免 label 超过图片本身。
        PARTICLE_BINARY_PANEL_LABEL_SIZE = 5.5
        # TBL 颗粒 full-frame 按用户最新要求使用纵向论文排版：
        # 1）SR/误差对比图每行两列，第一列是颗粒超分辨图，第二列是对应误差图；
        # 2）颗粒统计里的颗粒图/阈值图每行两列，第一列是颗粒超分辨图，第二列是阈值图。
        TBL_FULL_FRAME_VERTICAL_LAYOUT = True
        TBL_PARTICLE_STATS_IMAGE_VERTICAL_LAYOUT = True
        TBL_ERROR_MAP_PARTICLE_PAIR_LAYOUT = True
        # TBL full-frame 的 02_error_maps/particle_*_error.png 只保留 GT/SR 颗粒图和 crop 红框；
        # 第二列误差图及其色条关闭，避免误差图大面积空白和红框/label 混在一起。
        TBL_ERROR_MAP_PARTICLE_FULL_FRAME_IMAGE_ONLY = True
        # 该图的 label 放到右上角，避开左上角附近的 crop 红框和长实验名。
        TBL_ERROR_MAP_PARTICLE_LABEL_LOC = "upper_right"
        TBL_FULL_FRAME_ERROR_FILL_PANEL = True
        TBL_FULL_FRAME_FIG_WIDTH = 10.5
        TBL_FULL_FRAME_ROW_HEIGHT = 1.85
        TBL_FULL_FRAME_WSPACE = 0.08
        TBL_FULL_FRAME_HSPACE = 0.10
        # TBL full-frame 颗粒图/阈值图是“每行两列”的长条图像，方法名和 binary 标签比较长；
        # 这里把整张图宽度和每行高度加大，让每个小图有足够宽高，避免标题文字伸出图像边界。
        TBL_STATS_IMAGE_VERTICAL_FIG_WIDTH = 14.5
        TBL_STATS_IMAGE_VERTICAL_ROW_HEIGHT = 1.55
        # 颗粒阈值图横向和纵向间隔保持一致，并继续收紧，避免多列图之间出现大块空白。
        TBL_STATS_IMAGE_VERTICAL_HSPACE = 0.04
        TBL_STATS_IMAGE_VERTICAL_WSPACE = 0.04
        TBL_ERROR_MAP_VERTICAL_FIG_WIDTH = 10.5
        TBL_ERROR_MAP_VERTICAL_ROW_HEIGHT = 1.15
        TBL_ERROR_MAP_VERTICAL_HSPACE = 0.06
        TBL_ERROR_MAP_VERTICAL_WSPACE = 0.05
        # TBL 光流误差图底部存在由无效壁面/填充值形成的统一矩形带；绘图前自动裁掉这部分。
        # 这里只改论文图显示，不修改原始 npy 数据；阈值偏保守，必须从底部连续出现近似常值行才裁剪。
        TBL_FLOW_ERROR_TRIM_BOTTOM_ENABLED = True
        TBL_FLOW_ERROR_TRIM_MAX_FRACTION = 0.28
        TBL_FLOW_ERROR_TRIM_MIN_ROWS = 4
        TBL_FLOW_ERROR_TRIM_STD_RATIO = 0.18
        TBL_FLOW_ERROR_TRIM_MEAN_RATIO = 0.20
        # TBL full-frame 光流误差即使排除了 bicubic-widim/bicubic-hs，也可能被参考实验中的极少数尖峰像素撑大色条。
        # 因此 TBL 光流误差色条默认按参考实验绝对误差的分位数做 0 居中对称范围；设为 None 可恢复原始 min/max。
        TBL_FLOW_ERROR_COLORBAR_PERCENTILE = 99.5
        # TBL 颗粒 crop 框与原 evaluate 代码保持一致：框选区域 256x256，横向中心比例默认 0.265。
        # 颗粒 SR crop 没有单独 npy 时，pipeline.py 会用这里的 crop 框直接从原 LR/GT/SR npy 中裁；
        # 误差、二值图、统计 hist/stats 若存在 *_crop*.npy，则优先直接读取这些已有 crop 文件。
        TBL_PARTICLE_CROP_ENABLED = True
        TBL_PARTICLE_CROP_SIZE = 256
        TBL_PARTICLE_CROP_CENTER_RATIO = 0.265
        TBL_PARTICLE_CROP_OUTPUT_SUFFIX = "_crop"
        # TBL full-frame 颗粒 02_error_maps 中用于标出 crop 区域的红色正方形框。
        TBL_PARTICLE_CROP_BOX_COLOR = "red"
        TBL_PARTICLE_CROP_BOX_LINE_WIDTH = 1.3
        TBL_PARTICLE_CROP_FILE_NAMES = {
            "error": "{prefix}_sr_error_crop.npy",
            "hist": "{prefix}_particle_binary_stats_crop_hist.npy",
            "gt_binary": "{prefix}_particle_binary_stats_crop_gt_binary.npy",
            "sr_binary": "{prefix}_particle_binary_stats_crop_pred_binary.npy",
            "stats": "{prefix}_particle_binary_stats_crop_stats.npy",
        }
        # TBL u/v 光流面板：第一行加入 GT，只在第一行显示 u/v 字符，并且每行末尾都放一个色条。
        TBL_FLOW_UV_INCLUDE_GT = True
        TBL_FLOW_UV_COMPONENT_LABELS_FIRST_ROW_ONLY = True
        TBL_FLOW_UV_ROW_COLORBAR = True
        TBL_FLOW_UV_FIG_WIDTH = 6.0
        TBL_FLOW_UV_ROW_HEIGHT = 1.55
        # 八实验对比图如果全部横排会太挤；这类“数值图一行 + 误差图一行”的组合图，
        # 在 eight_experiments 中每块只放 4 个实验，后 4 个实验换到下一块，并从 GT 后面的第二列开始。
        COMPARISON_GROUP_COMPOSITE_WRAP_METHOD_COUNT = {
            "eight_experiments": 4,
        }
        # 颗粒阈值对比图包含 GT 和 LR 两个固定列；八组图后半部分从第三列开始对齐实验列。
        # 去掉 widim/hs 的六组补充图按 3+3 分块，同样从第三列开始。
        COMPARISON_GROUP_PARTICLE_BINARY_WRAP_METHOD_COUNT = {
            "eight_experiments": 4,
            "eight_experiments_without_widim_hs": 3,
        }
        # 指标汇总表列名全部集中到全局变量，后续如果论文表格需要改列名，只需要改这里。
        METRIC_TABLE_COLUMNS = (
            "comparison",
            "class",
            "split",
            "category",
            "experiment",
            "AEE",
            "NORM_AEE",
            "C_AEE",
            "MSE",
            "ES",
            "NRMSE",
            "PSNR",
            "SSIM",
        )
        # ALL_CLASS_flow.csv 中需要抽取的光流指标。部分历史文件列名略有差异，所以每个指标保留多个别名。
        METRIC_FLOW_FIELD_ALIASES = {
            "AEE": ("VAL_AEE", "AEE", "epe"),
            "NORM_AEE": (
                "VAL_NORM_AEE",
                "VAL_NORM_AEE_PER100PIXEL",
                "NORM_AEE",
                "NORM_AEE_PER100PIXEL",
            ),
            "C_AEE": ("VAL_C_AEE", "C_AEE"),
        }
        # ALL_CLASS_IMAGE_PAIR.csv 中需要抽取的图像指标；energy_spectrum_mse 按用户要求统一输出为 ES。
        METRIC_IMAGE_FIELD_ALIASES = {
            "MSE": ("mse", "MSE"),
            "ES": ("energy_spectrum", "energy_spectrum_mse", "ES", "ESMSE"),
            "NRMSE": ("nrmse", "NRMSE"),
            "PSNR": ("PSNR", "psnr"),
            "SSIM": ("SSIM", "ssim"),
        }

        # =========================
        # 坐标轴范围、tick 与色条范围
        # =========================
        # ENERGY_SPECTRUM 用 log-log 显示；None 表示由当前数据自动决定范围。
        ENERGY_SPECTRUM_X_MIN = 1.0
        ENERGY_SPECTRUM_X_MAX = 200
        ENERGY_SPECTRUM_Y_MIN = None
        ENERGY_SPECTRUM_Y_MAX = None
        ENERGY_SPECTRUM_X_TICK_INTERVAL = None
        ENERGY_SPECTRUM_Y_TICK_INTERVAL = None
        # TBL/TWCF 的能量谱横坐标比普通方形样本更长；如果继续套用全局 X_MAX=200，
        # 右侧高波数区域会在二维图和 2D+3D 合图左侧二维图中被截断。
        # 因此仅对这些类别按当前可用数据自动扩展横轴，其他类别仍使用上方全局范围。
        ENERGY_SPECTRUM_DYNAMIC_X_LIMIT_CATEGORIES = ("tbl", "twcf")
        ENERGY_SPECTRUM_DYNAMIC_X_MARGIN_RATIO = 0.02
        # 能量谱的“2D + 局部 3D”合图中，3D 子图按各实验 log10(energy) 曲线差异自动取 x 范围。
        # 阈值越大，3D 越聚焦在差异最明显的波数段；阈值越小，保留范围越宽。
        ENERGY_FOCUS_X_LIMIT_SPREAD_RATIO = 0.18
        ENERGY_FOCUS_X_LIMIT_MARGIN_RATIO = 0.08
        ENERGY_FOCUS_X_LIMIT_MIN_SPAN_RATIO = 0.18

        # FLOW_ERROR_COLORBAR_LIMIT / PARTICLE_ERROR_COLORBAR_LIMIT 支持两种模式：
        # - "auto": pipeline.py 会按 ERROR_COLORBAR_REFERENCE_EXPERIMENT_KEYS 指定的实验求 min/max；
        # - (min, max): 手动固定色条范围，例如 (-0.5, 0.5)。
        FLOW_ERROR_COLORBAR_LIMIT = "auto"
        PARTICLE_ERROR_COLORBAR_LIMIT = "auto"
        # experiment 专用局部放大颗粒对比区域，格式为 HR/SR 坐标的相对
        # (x_center, y_center, width, height)。用于 all_handle 的 experiment 局部放大 composite。
        EXPERIMENT_PARTICLE_ZOOM_REGIONS = (
            (0.18, 0.38, 0.08, 0.10),
            (0.42, 0.45, 0.08, 0.10),
            (0.72, 0.42, 0.08, 0.10),
        )
        VORTICITY_ERROR_COLORBAR_LIMIT = "auto"
        FLOW_VALUE_COLORBAR_LIMIT = "auto"
        # 当 test_all 只保存了 uvs_compare.png、没有原始 fake_flo/hr_flo.npy 时，
        # 光流值图会从 PNG 裁剪成 RGB 面板；RGB 面板本身没有物理数值，无法自动计算色条范围。
        # 这里给 u/v/s 提供与原测试绘图一致的兜底色条范围，保证 TBL 的 flow_u/v/s_value_error_composite
        # 第一行最右侧也能显示光流值色条。普通类别使用 default，TBL/TWCF 使用各自固定范围。
        FLOW_VALUE_COMPONENT_FALLBACK_LIMITS = {
            "default": {
                "u": (-4.0, 4.0),
                "v": (-4.0, 4.0),
                "s": (0.0, 5.6568542495),
            },
            "tbl": {
                "u": (2.0, 8.0),
                "v": (-0.5, 0.5),
                "s": (2.0, 8.0156097709),
            },
            "twcf": {
                "u": (-2.0, 12.0),
                "v": (-1.0, 1.0),
                "s": (0.0, 12.0415945788),
            },
        }
        PARTICLE_VALUE_COLORBAR_LIMIT = "auto"
        VORTICITY_VALUE_COLORBAR_LIMIT = "auto"
        # 光流/颗粒误差图的自动色条范围只参考以下四个实验：
        # bicubic-raft、srgan-raft、esrgan-raft 和 ESRuRAFT-PIV，避免 widim/hs 的极端误差撑大色条。
        ERROR_COLORBAR_REFERENCE_EXPERIMENT_KEYS = (
            "bicubic_raft",
            "srgan_raft",
            "esrgan_raft",
            "PIV_A_Esrgan_v4",
        )
        # 不同对比组可以单独覆盖参考实验。八组对比严格使用上面的四个实验；
        # x4/x8 对比没有 bicubic/srgan/esrgan，因此使用 x4 与 x8 自身确定色条。
        COMPARISON_GROUP_ERROR_COLORBAR_REFERENCE_KEYS = {
            "eight_experiments": ERROR_COLORBAR_REFERENCE_EXPERIMENT_KEYS,
            "eight_experiments_without_widim_hs": ERROR_COLORBAR_REFERENCE_EXPERIMENT_KEYS,
            "scale_x4_x8": ("PIV_A_Esrgan_v4", "PIV_A_Esrgan_v_SCALE_8"),
        }
        # 涡度位移对比图的速度/位移场箭头配置；第一行涡度底图会叠加 fake_flo 的 quiver 箭头。
        VORTICITY_QUIVER_STRIDE = 16
        VORTICITY_QUIVER_COLOR = "black"
        VORTICITY_QUIVER_WIDTH = 0.0025
        VORTICITY_QUIVER_ALPHA = 0.85
        VORTICITY_QUIVER_SCALE = None
        VORTICITY_QUIVER_HEADWIDTH = 3
        VORTICITY_QUIVER_HEADLENGTH = 4

        # 直方图坐标轴；None 表示自动范围。TBL/TWCF 若需要独立范围，可在 *_CATEGORY_AXIS_LIMITS 中配置。
        FLOW_ERROR_HIST_X_MIN = None
        FLOW_ERROR_HIST_X_MAX = None
        FLOW_ERROR_HIST_Y_MIN = 0.0
        FLOW_ERROR_HIST_Y_MAX = None
        FLOW_U_HIST_X_MIN = None
        FLOW_U_HIST_X_MAX = None
        FLOW_U_HIST_Y_MIN = 0.0
        FLOW_U_HIST_Y_MAX = None
        EPE_HIST_X_MIN = 0.0
        EPE_HIST_X_MAX = None
        EPE_HIST_Y_MIN = 0.0
        EPE_HIST_Y_MAX = None
        PARTICLE_ERROR_HIST_X_MIN = None
        PARTICLE_ERROR_HIST_X_MAX = None
        PARTICLE_ERROR_HIST_Y_MIN = 0.0
        PARTICLE_ERROR_HIST_Y_MAX = None
        VORTICITY_ERROR_HIST_X_MIN = None
        VORTICITY_ERROR_HIST_X_MAX = None
        VORTICITY_ERROR_HIST_Y_MIN = 0.0
        VORTICITY_ERROR_HIST_Y_MAX = None

        # TBL/TWCF 大图误差分布通常比普通样本宽，单独放配置，避免影响其它类别。
        FLOW_ERROR_HIST_CATEGORY_AXIS_LIMITS = {
            "tbl": {"x_min": -15, "x_max": 15, "y_min": 0.0, "y_max": None},
            "twcf": {"x_min": -15, "x_max": 15, "y_min": 0.0, "y_max": None},
        }
        PARTICLE_ERROR_HIST_CATEGORY_AXIS_LIMITS = {
            "tbl": {"x_min": -1.0, "x_max": 1.0, "y_min": 0.0, "y_max": None},
            "twcf": {"x_min": -1.2, "x_max": 1.2, "y_min": 0.0, "y_max": None},
        }
        VORTICITY_ERROR_HIST_CATEGORY_AXIS_LIMITS = {
            "tbl": {"x_min": -15, "x_max": 15, "y_min": 0.0, "y_max": None},
            "twcf": {"x_min": -15, "x_max": 15, "y_min": 0.0, "y_max": None},
        }

        # =========================
        # NPY 文件名约定
        # =========================
        # 不同阶段保存的文件名略有差异，pipeline.py 会按这些候选名依次尝试。
        # ENERGY_SPECTRUM 默认用于比较“超分辨率图像”的频谱，所以必须优先读取 image_pair_*。
        # 之前把 flow_energy_spectrum_* 放在最前，会导致 bicubic-widim / bicubic-hs / bicubic-raft /
        # bicubic-searaft 读取到不同光流算法输出的频谱；这些实验虽然 SR 图像同为 bicubic，
        # 但后续光流算法不同，flow 频谱自然不一致。若后续确实需要光流频谱，可把 flow_* 候选移回最前。
        ENERGY_SPECTRUM_FILE_CANDIDATES = (
            "image_pair_energy_spectrum_pred_mean.npy",
            "energy_spectrum_pred.npy",
            "flow_energy_spectrum_pred_mean.npy",
            "flow_energy_spectrum_pred.npy",
        )
        # GT 也优先读取真实图像对频谱，保证 GT 与 SR 曲线属于同一种物理量/数据来源。
        ENERGY_SPECTRUM_GT_FILE_CANDIDATES = (
            "image_pair_energy_spectrum_gt_mean.npy",
            "energy_spectrum_gt.npy",
            "flow_energy_spectrum_gt_mean.npy",
            "flow_energy_spectrum_gt.npy",
        )
        FLOW_HIST_FILE_NAMES = {
            "u": "delta_u_hist_all.npy",
            "v": "delta_v_hist_all.npy",
            "w": "delta_w_hist_all.npy",
            "epe": "epe_hist_all.npy",
        }
        FLOW_SAMPLE_HIST_FILE_NAMES = {
            "u": "delta_u_hist.npy",
            "v": "delta_v_hist.npy",
            "w": "delta_w_hist.npy",
            "epe": "epe_hist.npy",
        }
        PARTICLE_HIST_FILE_NAME = "sr_error_hist_all.npy"
        PARTICLE_SAMPLE_HIST_FILE_NAME = "sr_error_hist.npy"
        VORTICITY_HIST_FILE_NAME = "delta_vorticity_hist_all.npy"
        VORTICITY_SAMPLE_HIST_FILE_NAME = "delta_vorticity_hist.npy"

        FLOW_ARRAY_FILE_NAMES = {
            "pred": "fake_flo.npy",
            "gt": "hr_flo.npy",
            "delta_u": "delta_u.npy",
            "delta_v": "delta_v.npy",
            "delta_w": "delta_w.npy",
        }
        # test_all 中有些样本没有 fake_flo/hr_flo 原始 npy，只保留了已经渲染好的 uvs_compare.png。
        # 这里把 PNG 文件名和裁剪参数放到全局变量：pipeline.py 会从该图中裁出 Pred U/V/S 与 GT U/V/S，
        # 避免光流组合图左侧 GT 面板为空白；如果以后原图布局变化，只需要微调这些阈值。
        FLOW_UVS_COMPARE_FILE_NAME = "uvs_compare.png"
        FLOW_UVS_COMPARE_ROW_COMPONENTS = ("u", "v", "s")
        FLOW_UVS_COMPARE_VALUE_COLUMNS = {
            "pred": 0,
            "gt": 1,
        }
        FLOW_UVS_COMPARE_TOTAL_COLUMNS = 3
        FLOW_UVS_COMPARE_WHITE_THRESHOLD = 0.985
        FLOW_UVS_COMPARE_ROW_MASK_FRACTION = 0.25
        FLOW_UVS_COMPARE_COL_MASK_FRACTION = 0.35
        FLOW_UVS_COMPARE_MIN_PANEL_FRACTION = 0.25
        FLOW_UVS_COMPARE_FALLBACK_CROP_RATIOS = {
            "left": 0.02,
            "right": 0.82,
            "top": 0.09,
            "bottom": 0.96,
        }
        # TBL 剖面文件名约定：{component} 会被替换为 u 或 v。
        # profile_columns.npy 是每个剖面所在的 x 列位置，profile_y_positions.npy 是纵向采样坐标。
        TBL_PROFILE_DIR_NAME = "profile_analysis"
        TBL_PROFILE_COMPONENTS = ("u", "v")
        TBL_PROFILE_FILE_NAMES = {
            "pred": "{component}_profile_pred.npy",
            "gt": "{component}_profile_gt.npy",
            "y": "profile_y_positions.npy",
            "columns": "profile_columns.npy",
        }
        PARTICLE_ARRAY_FILE_NAMES = {
            "lr": "lr.npy",
            "gt": "hr.npy",
            "sr": "fake.npy",
            "error": "sr_error.npy",
            "hist": "particle_binary_stats_hist.npy",
            "gt_binary": "particle_binary_stats_gt_binary.npy",
            "sr_binary": "particle_binary_stats_pred_binary.npy",
            "stats": "particle_binary_stats_stats.npy",
            "threshold": "particle_binary_stats_threshold.txt",
        }
        PARTICLE_IMAGE_FALLBACK_NAMES = {
            "lr": "lr.png",
            "gt": "hr.png",
            "sr": "fake.png",
        }
        VORTICITY_ARRAY_FILE_NAMES = {
            "pred": "pred_vorticity.npy",
            "gt": "gt_vorticity.npy",
            "error": "delta_vorticity.npy",
        }

        # 颗粒统计 npy/csv 中常见字段名；读取失败时 pipeline.py 会按位置兜底。
        PARTICLE_STATS_FIELD_ALIASES = {
            # 颗粒统计文件实际常以 metric/value 两列表保存，实验对比应使用 SR/预测颗粒的统计量，
            # 所以 count 和 pixels 优先映射 pred_* 字段，而不是 gt_* 或图像尺寸字段。
            "count": ("pred_particle_count", "pred_count", "count", "num_particles", "particle_count"),
            "pixels": ("pred_particle_pixels", "pred_pixels", "particle_pixels", "pixel_count", "area"),
            "mean_area": (
                "pred_area_mean",
                "pred_mean_area",
                "pred_particle_mean_area",
                "mean_area",
                "particle_mean_area",
                "avg_area",
                "average_area",
                "mean_particle_area",
            ),
            # IoU / precision / recall / F1 是预测阈值图与 GT 阈值图的二值重叠指标，实际字段带 binary_ 前缀。
            "iou": ("binary_iou", "iou", "IoU"),
            "precision": ("binary_precision", "precision", "Precision"),
            "recall": ("binary_recall", "recall", "Recall"),
            "f1": ("binary_f1", "f1", "F1", "f1_score"),
        }
        # 颗粒统计条形图加入 GT：count/pixels 读取 gt_* 字段；二值重叠指标是 GT 对 GT，理论值为 1。
        PARTICLE_GT_STATS_FIELD_ALIASES = {
            "count": ("gt_particle_count", "gt_count", "gt_num_particles", "gt_particle_count"),
            "pixels": ("gt_particle_pixels", "gt_pixels", "gt_pixel_count", "gt_area"),
            "mean_area": (
                "gt_area_mean",
                "gt_area_meanton",
                "gt_mean_area",
                "gt_particle_mean_area",
                "gt_avg_area",
                "gt_average_area",
                "gt_mean_particle_area",
            ),
        }
        PARTICLE_GT_SELF_METRIC_VALUE = 1.0
