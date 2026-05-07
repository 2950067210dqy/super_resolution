from study.SRGAN.model.ESRuRAFT_PIV.global_class import global_data
from study.SRGAN.model.tfrecord_test_common import run_test_all


def test_all(model, class_name, data_type, SCALE, device=None):
    """
    ESRuRAFT_PIV 分支的 TFRecord 全数据集测试入口。

    具体实现放在公共 run_test_all 中，保证四个分支的 dataset、下采样、
    滑窗拼接、指标与图片保存行为保持一致。
    """
    return run_test_all(
        model=model,
        global_data=global_data,
        class_name=class_name,
        data_type=data_type,
        SCALE=SCALE,
        device=device,
    )
