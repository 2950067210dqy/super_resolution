import os
import traceback

from loguru import logger

from study.SRGAN.model.PIV_esrgan_RAFT.Module.PIV_ESRGAN_RAFT_Model import PIV_ESRGAN_RAFT
from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data
from study.SRGAN.model.PIV_esrgan_RAFT.test import test_all
from study.SRGAN.model.pipeline_test_common import run_pipeline_test


def _build_model(_scale):
    """
    创建 PIV_esrgan_RAFT 分支的测试模型。

    这里保持与训练 pipeline 一致，避免模型目录里的 checkpoint 和当前实例结构不匹配。
    """
    return PIV_ESRGAN_RAFT(
        inner_chanel=3,
        batch_size=global_data.esrgan.BATCH_SIZE,
    )


def _resolve_data_type():
    """
    PIV_esrgan_RAFT 训练/测试目录固定使用 RAFT 作为 data_type。
    """
    return "RAFT"


def main():
    """
    加载训练好的 PIV_esrgan_RAFT 模型，并对 TEST_DATASETS 执行完整 test_all。
    """
    return run_pipeline_test(
        global_data=global_data,
        branch_name="PIV_esrgan_RAFT",
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
