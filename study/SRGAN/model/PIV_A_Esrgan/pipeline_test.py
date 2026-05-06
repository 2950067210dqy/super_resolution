import os
import traceback

from loguru import logger

from study.SRGAN.model.PIV_A_Esrgan.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
from study.SRGAN.model.PIV_A_Esrgan.global_class import global_data
from study.SRGAN.model.PIV_A_Esrgan.test import test_all
from study.SRGAN.model.pipeline_test_common import run_pipeline_test


def _build_model(_scale):
    """
    创建 PIV_A_Esrgan 分支的测试模型。

    这里保持与训练 pipeline 完全一致：
        - inner_chanel=3
        - batch_size 复用全局变量
    """
    return ESRuRAFT_PIV(
        inner_chanel=3,
        batch_size=global_data.esrgan.BATCH_SIZE,
    )


def _resolve_data_type():
    """
    PIV_A_Esrgan 的输出目录名与训练阶段保持一致。

    当 USE_RAFT=True 时，模型目录位于 .../RAFT/...；
    当 USE_RAFT=False 时，模型目录位于 .../SR/...。
    """
    return "RAFT" if global_data.esrgan.USE_RAFT else "SR"


def main():
    """
    加载训练好的 PIV_A_Esrgan 模型，并对 TEST_DATASETS 执行完整 test_all。
    """

    return run_pipeline_test(
        global_data=global_data,
        branch_name="PIV_A_Esrgan",
        model_factory=_build_model,
        data_type_resolver=_resolve_data_type,
        test_all_fn=test_all,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"{e}\n{traceback.format_exc()}")
    finally:
        if global_data.esrgan.IS_AUTO_DL:
            os.system("/usr/bin/shutdown")
