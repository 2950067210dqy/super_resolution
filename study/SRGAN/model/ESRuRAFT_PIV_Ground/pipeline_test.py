import os
import traceback

from loguru import logger

from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
from study.SRGAN.model.ESRuRAFT_PIV_Ground.global_class import global_data
from study.SRGAN.model.ESRuRAFT_PIV_Ground.test import test_all
from study.SRGAN.model.pipeline_test_common import run_pipeline_test


def _build_model(scale):
    """
    创建 ESRuRAFT_PIV_Ground 分支的测试模型。

    Ground 分支的 ESRGAN/SRGAN 生成器结构会受到当前 SCALE 影响，
    因此这里和训练 pipeline 一样把 sr_scale=SCALE 传进去，保证加载的 checkpoint
    与模型结构一一对应。
    """
    return ESRuRAFT_PIV(
        inner_chanel=3,
        batch_size=global_data.esrgan.BATCH_SIZE,
        sr_scale=scale,
    )


def _resolve_data_type():
    """
    ESRuRAFT_PIV_Ground 训练/测试目录固定使用 RAFT 作为 data_type。
    """
    return "RAFT"


def main():
    """
    加载训练好的 ESRuRAFT_PIV_Ground 模型，并对 TEST_DATASETS 执行完整 test_all。
    """
    return run_pipeline_test(
        global_data=global_data,
        branch_name="ESRuRAFT_PIV_Ground",
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
