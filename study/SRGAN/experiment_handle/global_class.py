from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern


@dataclass(frozen=True)
class ExperimentHandleConfig:
    INPUT_DIR: Path = Path(r"D:\BaiduSyncdisk\AYanJiuSheng\data\train_datas\experiment")
    OUTPUT_DIR: Path = Path(r"D:\BaiduSyncdisk\AYanJiuSheng\data\train_datas\experiment_handle")
    BACKGROUND_NAME: str = "bj.bmp"
    IMAGE_SUFFIXES: tuple[str, ...] = (".bmp",)

    GROUP_NAME_TEMPLATE: str = "exp_{index:04d}"
    IMG1_SUFFIX: str = "_img1.bmp"
    IMG2_SUFFIX: str = "_img2.bmp"
    FLOW_SUFFIX: str = "_flow.flo"
    FLOW_U_SUFFIX: str = "_flow_u.png"
    FLOW_V_SUFFIX: str = "_flow_v.png"

    FLO_MAGIC: float = 202021.25
    NATURAL_SORT_PATTERN: Pattern[str] = re.compile(r"(\d+)")

    GRAY_RGB_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)
    COMPONENT_PREVIEW_CENTER: int = 127
    COMPONENT_PREVIEW_SCALE: float = 127.5
    COMPONENT_PREVIEW_EPS: float = 1e-12

    OPENPIV_PACKAGE_NAME: str = "openpiv"
    OPENPIV_PROCESS_MODULE_NAME: str = "pyprocess"
    OPENPIV_PROCESS_FUNCTION_NAME: str = "extended_search_area_piv"
    OPENPIV_COORDINATE_FUNCTION_NAME: str = "get_coordinates"
    OPENPIV_WINDOW_SIZE: int = 32
    OPENPIV_OVERLAP: int = 16
    OPENPIV_SEARCH_AREA_SIZE: int = 64
    OPENPIV_DT: int = 1
    OPENPIV_SIG2NOISE_METHOD: str = "peak2peak"
    OPENPIV_USE_COORDINATES: bool = True
    OPENPIV_IMPORT_ERROR_MESSAGE: str = (
        "OpenPIV is not available. Install openpiv, then rerun this script."
    )


global_data = ExperimentHandleConfig()
