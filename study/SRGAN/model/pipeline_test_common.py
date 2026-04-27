from pathlib import Path

import torch
from loguru import logger

from study.SRGAN.data_load import filter_excluded_class_names, get_class_names


def _validate_train_class_mode(global_data):
    """
    统一读取 TRAIN_CLASS_MODE。

    大多数分支已经在 global_class 里提供 validate_train_class_mode()；
    这里优先复用它，避免 pipeline_test 和训练 pipeline 对模式字符串的判断出现偏差。
    """
    validator = getattr(global_data.esrgan, "validate_train_class_mode", None)
    if callable(validator):
        return validator()
    return str(getattr(global_data.esrgan, "TRAIN_CLASS_MODE", "mixed")).strip().lower()


def _select_single_class(available_class_names, preset_name=None):
    """
    复用训练 pipeline 的 single 选择语义。

    规则：
        - preset_name 非空且合法：直接使用；
        - 只有一个可选类别：直接返回；
        - 其他情况：保持与训练入口一致，允许手动输入索引。
    """
    if not available_class_names:
        raise ValueError("没有可用于 single 模式的类别。")

    if preset_name is not None:
        if preset_name not in available_class_names:
            raise ValueError(
                f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}"
            )
        return preset_name

    if len(available_class_names) == 1:
        return available_class_names[0]

    logger.info("TRAIN_CLASS_MODE='single' 且 SINGLE_CLASS_NAME=None，请选择一个类别：")
    for idx, class_name in enumerate(available_class_names):
        logger.info(f"  [{idx}] {class_name}")

    while True:
        answer = input("请输入类别索引: ").strip()
        try:
            index = int(answer)
            if 0 <= index < len(available_class_names):
                return available_class_names[index]
        except Exception:
            pass
        logger.warning("输入无效，请重新输入可用索引。")


def _build_run_jobs(global_data):
    """
    按当前 TRAIN_CLASS_MODE 构造要测试的任务列表。

    这里刻意复用训练 pipeline 的任务拆分规则：
        - all: 每个类别分别测；
        - single: 只测一个类别；
        - mixed: 用 mixed_all_classes 这个聚合任务目录；
        - fixed: 用 fixed_train_validate 这个固定划分任务目录。
    """
    available_class_names = get_class_names(global_data.esrgan.GR_DATA_ROOT_DIR)
    logger.info(f"[pipeline_test] 一共{len(available_class_names)}个类别：{available_class_names}")

    mode = _validate_train_class_mode(global_data)
    if mode == "fixed":
        # fixed 模式的比例超参数是由 list 文件反推出来的；这里同步一次，
        # 主要是为了让 pipeline_test 的日志/超参数文本与训练阶段口径一致。
        updater = getattr(global_data.esrgan, "update_fixed_split_rates", None)
        if callable(updater):
            updater()

    if mode in {"all", "mixed", "fixed"}:
        available_class_names = filter_excluded_class_names(
            available_class_names,
            getattr(global_data.esrgan, "EXCLUDE_CLASS", None),
            context=f"{global_data.esrgan.name}:pipeline_test:{mode}",
        )
        logger.info(f"[pipeline_test] 排除类别后剩余{len(available_class_names)}个类别：{available_class_names}")

    run_jobs = []
    if mode == "all":
        for class_name in available_class_names:
            run_jobs.append({"run_class_name": class_name, "selected_classes": [class_name]})
    elif mode == "single":
        chosen = _select_single_class(available_class_names, getattr(global_data.esrgan, "SINGLE_CLASS_NAME", None))
        run_jobs.append({"run_class_name": chosen, "selected_classes": [chosen]})
    elif mode == "mixed":
        run_jobs.append({"run_class_name": global_data.esrgan.MIXED_CLASS_TAG, "selected_classes": None})
    elif mode == "fixed":
        run_jobs.append({"run_class_name": global_data.esrgan.FIXED_CLASS_TAG, "selected_classes": None})
    else:
        raise ValueError(f"Unsupported TRAIN_CLASS_MODE: {mode}")

    return mode, run_jobs


def _resolve_model_dir(global_data, class_name, data_type, scale):
    """
    统一拼接训练权重目录。

    注意 MODEL_DIR 在全局变量里通常写成 '/train_model'，因此这里先 strip 掉开头的分隔符，
    再用 Path 组合，避免 Windows/Linux 下出现奇怪的绝对路径覆盖行为。
    """
    scale_dir = Path(
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(scale * scale)}"
    )
    model_dir_name = str(global_data.esrgan.MODEL_DIR).strip("/\\")
    return scale_dir / model_dir_name


def _find_checkpoint_path(global_data, class_name, data_type, scale):
    """
    根据用户要求的通配符规则寻找模型权重。

    匹配规则：
        f\".../{MODEL_DIR}/*_model_{global_data.esrgan.name}.pth\"

    如果匹配到多个文件，则优先选择“最近修改”的那个，并把所有候选写入日志，
    方便你排查同一目录下存在多个模型文件时到底用了哪一个。
    """
    model_dir = _resolve_model_dir(global_data, class_name, data_type, scale)
    pattern = f"*_model_{global_data.esrgan.name}.pth"
    matches = sorted(model_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"[pipeline_test] 未找到模型文件，匹配规则为: {model_dir / pattern}"
        )
    if len(matches) > 1:
        logger.warning(
            "[pipeline_test] 找到多个模型文件，默认使用最新一个：\n{}",
            "\n".join([str(path) for path in matches]),
        )
    return matches[0]


def _extract_model_state_dict(checkpoint_obj):
    """
    从 torch.load 的结果中提取真正的模型 state_dict。

    兼容两种常见保存方式：
        1. 直接 torch.save(model.state_dict(), path)
        2. torch.save({'model_state_dict': ...}, path)
    """
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    return checkpoint_obj


def _normalize_state_dict_keys_for_model(state_dict, model):
    """
    兼容 DataParallel/非 DataParallel 的 key 前缀差异。

    如果 checkpoint 里是 module.xxx，而当前单卡测试模型是 xxx，这里自动去掉 module.；
    反过来如果当前模型真的带 module. 前缀，则保留原样。
    """
    if not isinstance(state_dict, dict):
        return state_dict

    model_keys = list(model.state_dict().keys())
    has_model_module_prefix = any(key.startswith("module.") for key in model_keys)
    has_state_module_prefix = any(str(key).startswith("module.") for key in state_dict.keys())
    if has_state_module_prefix and not has_model_module_prefix:
        return {
            (key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def run_pipeline_test(global_data, branch_name, model_factory, data_type_resolver, test_all_fn):
    """
    四个分支共享的 pipeline_test 主流程。

    功能：
        1. 按 TRAIN_CLASS_MODE / EXCLUDE_CLASS / SCALES 枚举测试任务；
        2. 在对应输出目录下按用户要求的通配符寻找训练好的模型；
        3. 加载模型权重；
        4. 强制开启 IS_TEST，并调用各分支 test_all 做完整测试。
    """
    device = getattr(global_data.esrgan, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _, run_jobs = _build_run_jobs(global_data)
    original_is_test = getattr(global_data.esrgan, "IS_TEST", False)
    results = []

    try:
        # pipeline_test 是一个专门的“只做 test_all”的入口，因此这里临时强制打开 IS_TEST。
        # 这样不会改变 global_class 里的默认值，只影响当前这次脚本运行。
        global_data.esrgan.IS_TEST = True

        for job in run_jobs:
            class_name = job["run_class_name"]
            data_type = str(data_type_resolver()).strip()

            for scale in getattr(global_data.esrgan, "SCALES", [1]):
                model = model_factory(scale).to(device, non_blocking=(getattr(device, "type", "cpu") == "cuda"))
                checkpoint_path = _find_checkpoint_path(global_data, class_name, data_type, scale)
                checkpoint_obj = torch.load(checkpoint_path, map_location=device)
                state_dict = _extract_model_state_dict(checkpoint_obj)
                state_dict = _normalize_state_dict_keys_for_model(state_dict, model)
                model.load_state_dict(state_dict)

                logger.info(
                    "[pipeline_test] branch={} | class_name={} | data_type={} | scale={} | checkpoint={}",
                    branch_name,
                    class_name,
                    data_type,
                    scale,
                    checkpoint_path,
                )

                summary_rows = test_all_fn(
                    model=model,
                    class_name=class_name,
                    data_type=data_type,
                    SCALE=scale,
                    device=device,
                )
                results.append(
                    {
                        "branch": branch_name,
                        "class_name": class_name,
                        "data_type": data_type,
                        "scale": scale,
                        "checkpoint_path": str(checkpoint_path),
                        "summary_rows": summary_rows,
                    }
                )
        return results
    finally:
        global_data.esrgan.IS_TEST = original_is_test
