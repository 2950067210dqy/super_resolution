from study.SRGAN.model.PIV_A_Esrgan.global_class import global_data
from study.SRGAN.model.tfrecord_test_common import run_test_all


def test_all(model, class_name, data_type, SCALE, device=None):
    """
    PIV_A_Esrgan 分支的 TFRecord 全数据集测试入口。

    具体实现放在公共 run_test_all 中，保证 PIV_A_Esrgan、ESRuRAFT_PIV、
    ESRuRAFT_PIV_Ground、PIV_esrgan_RAFT 四个分支的测试逻辑完全一致。
    """
    return run_test_all(
        model=model,
        global_data=global_data,
        class_name=class_name,
        data_type=data_type,
        SCALE=SCALE,
        device=device,
    )
