from study.SRGAN.model.ESRuRAFT_PIV_Ground.global_class import global_data
from study.SRGAN.model.tfrecord_test_common import run_test_all


def test_all(model, class_name, data_type, SCALE, device=None):
    """
    ESRuRAFT_PIV_Ground 分支的 TFRecord 全数据集测试入口。

    Ground/SR/RAFT 模式的差异仍由 PIV_ESRGAN_RAFT_Model 内部处理；这里仅负责
    统一读取 RAFT256-PIV 风格 TFRecord 并触发联合模型测试。
    """
    return run_test_all(
        model=model,
        global_data=global_data,
        class_name=class_name,
        data_type=data_type,
        SCALE=SCALE,
        device=device,
    )
