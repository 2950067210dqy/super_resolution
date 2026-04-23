from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data
from study.SRGAN.model.tfrecord_test_common import run_test_all


def test_all(model, class_name, data_type, SCALE, device=None):
    """
    PIV_esrgan_RAFT 分支的 TFRecord 全数据集测试入口。

    具体实现放在公共 run_test_all 中，避免四个分支出现测试集路径、
    滑窗推理或指标保存格式不一致的问题。
    """
    return run_test_all(
        model=model,
        global_data=global_data,
        class_name=class_name,
        data_type=data_type,
        SCALE=SCALE,
        device=device,
    )
