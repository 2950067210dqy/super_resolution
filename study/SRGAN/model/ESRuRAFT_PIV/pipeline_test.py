from study.SRGAN.model.ESRuRAFT_PIV.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
from study.SRGAN.model.ESRuRAFT_PIV.global_class import global_data
from study.SRGAN.model.ESRuRAFT_PIV.test import test_all
from study.SRGAN.model.pipeline_test_common import run_pipeline_test


def _build_model(_scale):
    """
    创建 ESRuRAFT_PIV 分支的测试模型。

    测试入口严格复用训练时的模型构造参数，避免因为测试脚本和训练脚本实例化不一致，
    导致 checkpoint 能加载但推理行为不一致。
    """
    return ESRuRAFT_PIV(
        inner_chanel=3,
        batch_size=global_data.esrgan.BATCH_SIZE,
    )


def _resolve_data_type():
    """
    ESRuRAFT_PIV 训练/测试目录固定使用 RAFT 作为 data_type。
    """
    return "RAFT"


def main():
    """
    加载训练好的 ESRuRAFT_PIV 模型，并对 TEST_DATASETS 执行完整 test_all。
    """
    return run_pipeline_test(
        global_data=global_data,
        branch_name="ESRuRAFT_PIV",
        model_factory=_build_model,
        data_type_resolver=_resolve_data_type,
        test_all_fn=test_all,
    )


if __name__ == "__main__":
    main()
