"""
compute_cost.py

这个脚本的目标是做“模型计算开销分析”，常见输出包括：
1. 参数量（parameters）
2. FLOPs / 浮点运算次数
3. 单次推理延迟（latency）
4. 吞吐量（throughput）
5. GPU 峰值显存（peak GPU memory）
6. 将结果保存为 txt / csv

当前这个文件更像一个“通用 profiling 工具模块”：
- 前半部分提供一组可复用函数
- 后半部分负责解析命令行、组织结果、保存输出

注意：
- 这里的 FLOPs 依赖 `torch.profiler` 的统计结果，本质是“估算值”
- GPU 峰值显存只在 CUDA 下有效
- 如果模型 forward 需要多个输入，可以使用 `*_for_inputs` 这一组函数
"""

# argparse: 用于解析命令行参数，例如 --batch-size 32。
import argparse
# csv: 用于把 profiling 结果保存成结构化 CSV 文件。
import csv
# sys: 主要用于动态修改 sys.path，让脚本能导入项目里的模块。
import sys
# time: 负责时间戳生成，以及 latency 测量。
import time
# Path: 更安全、清晰地处理文件路径。
from pathlib import Path

# torch: Profiling 的核心依赖，参数量 / FLOPs / 延迟 / 显存都依赖它。
import torch


# 默认项目根目录。
# 这里假设项目结构里 `study` 目录下面包含 `SRGAN` 包。
DEFAULT_PROJECT_ROOT = Path(r"D:\WorkSpace\super_reloution_project\study")


def parse_args():
    """
    解析命令行参数。

    这个函数的职责只有一个：
    把命令行输入整理成一个 `args` 对象，后续主流程直接读取即可。

    返回：
        argparse.Namespace:
            一个包含全部配置项的对象，例如：
            - args.batch_size
            - args.device
            - args.output_dir
    """
    # 创建 ArgumentParser 实例，并写入脚本用途说明。
    parser = argparse.ArgumentParser(
        description="Profile ESRGAN generator/discriminator: params, FLOPs, latency, throughput, and GPU memory."
    )
    # project-root: 项目根路径，用于后续导入项目内模块。
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root that contains the SRGAN package.",
    )
    # generator-checkpoint: 可选，给生成器加载预训练权重。
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=None,
        help="Optional generator checkpoint (.pth) to load before profiling.",
    )
    # discriminator-checkpoint: 可选，给判别器加载预训练权重。
    parser.add_argument(
        "--discriminator-checkpoint",
        type=Path,
        default=None,
        help="Optional discriminator checkpoint (.pth) to load before profiling.",
    )
    # batch-size: profiling 时输入 batch 的大小。
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    # channels: 输入图像通道数，例如 RGB 通常是 3。
    parser.add_argument("--channels", type=int, default=3, help="Input image channels.")
    # height: 低分辨率输入的高。
    parser.add_argument("--height", type=int, default=64, help="LR input height.")
    # width: 低分辨率输入的宽。
    parser.add_argument("--width", type=int, default=64, help="LR input width.")
    # num-blocks: ESRGAN 生成器内部 RRDB block 的数量。
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=23,
        help="Number of RRDB blocks used by Generator.",
    )
    # scale: 上采样倍率。
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Internal upsample factor used by each upsample stage in Generator.",
    )
    # device: 指定运行设备。
    # auto 表示自动选；cuda 表示强制 GPU；cpu 表示强制 CPU。
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run profiling on.",
    )
    # warmup: 正式计时前先预热多少轮，减少首次运行带来的偏差。
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    # iters: 正式测量多少轮，轮数越多统计越稳定。
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
    # output-dir: txt/csv 结果保存目录。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "profile_results",
        help="Directory used to save txt/csv profiling results.",
    )
    # tag: 实验标签，用于输出文件命名，也会写入 CSV。
    parser.add_argument(
        "--tag",
        type=str,
        default="esrgan_profile",
        help="Experiment tag used in saved filenames and CSV rows.",
    )
    # 解析并返回全部参数。
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """
    根据命令行参数解析最终使用的设备。

    参数：
        device_arg: str
            用户输入的设备选项，取值只能是 auto / cuda / cpu。

    返回：
        torch.device:
            最终用于 profiling 的设备对象。

    逻辑：
    - cuda: 必须真的有 GPU，否则直接报错
    - cpu: 直接返回 CPU
    - auto: 有 GPU 就用 GPU，否则退回 CPU
    """
    # 如果用户明确要求 cuda，就必须校验 CUDA 是否真的可用。
    if device_arg == "cuda":
        # 如果机器上不可用 CUDA，则直接抛异常，避免“以为在测 GPU，实际跑的是 CPU”。
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --device cuda was requested.")
        # CUDA 可用时，返回 GPU 设备。
        return torch.device("cuda")
    # 如果用户明确要求 CPU，则直接返回 CPU。
    if device_arg == "cpu":
        return torch.device("cpu")
    # auto 模式：优先 GPU，不行就 CPU。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    """
    可选地为模型加载 checkpoint。

    参数：
        model:
            要加载权重的模型。
        checkpoint_path:
            权重路径；如果是 None，表示不加载。
        device:
            权重加载到哪个设备上。

    返回：
        str:
            返回实际使用的 checkpoint 路径字符串；
            如果没加载，则返回 "None"。
    """
    # 如果没有传 checkpoint，就直接返回字符串 "None"，方便写入结果表。
    if checkpoint_path is None:
        return "None"
    # 如果传了路径，但文件不存在，则立即报错。
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # 按指定设备加载权重。
    state = torch.load(checkpoint_path, map_location=device)
    # 有些训练脚本会把真正参数放在 state_dict 字段里，这里做兼容处理。
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # 严格加载参数；如果 key 不匹配，让错误尽早暴露。
    model.load_state_dict(state, strict=True)
    # 返回路径字符串，便于打印和写入结果文件。
    return str(checkpoint_path)


def count_parameters(model: torch.nn.Module):
    """
    统计模型参数量。

    返回：
        tuple[int, int]:
            - total: 全部参数量
            - trainable: 可训练参数量（requires_grad=True）
    """
    # total: 遍历所有参数张量，累加其元素个数。
    total = sum(p.numel() for p in model.parameters())
    # trainable: 只统计参与梯度更新的参数。
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 返回总参数量和可训练参数量。
    return total, trainable


def format_count(value: float) -> str:
    """
    把数字格式化成更适合人读的单位字符串。

    示例：
        1234 -> "1.234 K"
        1200000 -> "1.200 M"
        2500000000 -> "2.500 G"
    """
    # 大于等于 1e9 时，用 G 表示。
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    # 大于等于 1e6 时，用 M 表示。
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    # 大于等于 1e3 时，用 K 表示。
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    # 更小的数直接保留整数形式。
    return f"{value:.0f}"


def estimate_flops(model: torch.nn.Module, sample: torch.Tensor, device: torch.device):
    """
    对“单输入模型”估算 FLOPs。

    参数：
        model:
            待分析模型。
        sample:
            单个输入样本张量。
        device:
            当前运行设备。

    返回：
        int | float:
            累加得到的 FLOPs 估算值。
    """
    # 默认至少统计 CPU 活动。
    activities = [torch.profiler.ProfilerActivity.CPU]
    # 如果运行在 CUDA 上，就把 GPU 活动也纳入 profiler。
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Profiling 不需要梯度，因此关闭 autograd，减少额外开销。
    with torch.no_grad():
        # 开启 profiler，并要求输出 FLOPs 信息。
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            # 跑一次 forward，profiler 会记录各算子的 FLOPs。
            _ = model(sample)

    # total_flops: 用于累加所有算子的 FLOPs。
    total_flops = 0
    # prof.key_averages() 会给出聚合后的事件列表。
    for event in prof.key_averages():
        # 某些事件可能没有 flops 字段，因此用 getattr 做兼容。
        event_flops = getattr(event, "flops", 0)
        # 只累加非空值。
        if event_flops is not None:
            total_flops += event_flops
    # 返回总 FLOPs。
    return total_flops


def estimate_flops_for_inputs(model: torch.nn.Module, inputs: tuple, device: torch.device):
    """
    对“多输入模型”估算 FLOPs。

    适用场景：
    - model(x1, x2)
    - model(img_prev, img_next, flow_gt)
    - 任何 forward 需要多个位置参数的情况
    """
    # 默认先统计 CPU 活动。
    activities = [torch.profiler.ProfilerActivity.CPU]
    # 如果是 CUDA 设备，则追加 GPU 活动统计。
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # 关闭梯度，保证 profiling 更接近纯推理成本。
    with torch.no_grad():
        # 开启 profiler，并启用 FLOPs 统计。
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            # 这里用 *inputs 展开多个输入。
            _ = model(*inputs)

    # 初始化 FLOPs 总和。
    total_flops = 0
    # 遍历 profiler 汇总后的事件。
    for event in prof.key_averages():
        # 尝试取出 event.flops；如果没有则给 0。
        event_flops = getattr(event, "flops", 0)
        # 累加有效 FLOPs。
        if event_flops is not None:
            total_flops += event_flops
    # 返回汇总 FLOPs。
    return total_flops


def benchmark_latency(model: torch.nn.Module, sample: torch.Tensor, device: torch.device, warmup: int, iters: int):
    """
    测量“单输入模型”的平均推理延迟和吞吐量。

    返回：
        tuple[float, float]:
            - latency_ms: 平均每个 batch 的毫秒耗时
            - throughput: 每秒处理多少张图
    """
    # 推理计时不需要梯度。
    with torch.no_grad():
        # 先做 warmup，让 cudnn kernel 选择、缓存初始化等开销提前发生。
        for _ in range(warmup):
            _ = model(sample)
        # CUDA 是异步执行的，正式计时前先同步一次，避免 warmup 残留影响结果。
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 记录开始时间。
        start = time.perf_counter()
        # 正式执行 iters 轮 forward。
        for _ in range(iters):
            _ = model(sample)
        # 再同步一次，确保 GPU 真的全部执行完成。
        if device.type == "cuda":
            torch.cuda.synchronize()
        # 记录总耗时。
        elapsed = time.perf_counter() - start

    # 平均单次 batch 的毫秒延迟。
    latency_ms = elapsed * 1000.0 / iters
    # 吞吐量 = 总处理图片数 / 总耗时。
    throughput = iters * sample.shape[0] / elapsed
    # 返回延迟和吞吐量。
    return latency_ms, throughput


def benchmark_latency_for_inputs(model: torch.nn.Module, inputs: tuple, device: torch.device, warmup: int, iters: int):
    """
    测量“多输入模型”的平均推理延迟和吞吐量。

    约定：
    - 默认把 `inputs[0]` 看成主输入
    - 吞吐量中的 batch_size 从 `inputs[0].shape[0]` 读取
    """
    # 如果 inputs[0] 有 shape，就把它的第 0 维视为 batch_size；否则默认按 1 处理。
    batch_size = inputs[0].shape[0] if inputs and hasattr(inputs[0], "shape") else 1

    # 关闭梯度，只测推理路径。
    with torch.no_grad():
        # 先预热 warmup 轮。
        for _ in range(warmup):
            _ = model(*inputs)
        # 如果在 GPU 上跑，计时前先同步。
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 正式开始计时。
        start = time.perf_counter()
        # 连续执行 iters 轮 forward。
        for _ in range(iters):
            _ = model(*inputs)
        # GPU 异步执行，因此结尾也需要同步。
        if device.type == "cuda":
            torch.cuda.synchronize()
        # 统计总耗时。
        elapsed = time.perf_counter() - start

    # 计算平均每次 batch 的毫秒延迟。
    latency_ms = elapsed * 1000.0 / iters
    # 计算每秒吞吐的样本数。
    throughput = iters * batch_size / elapsed
    # 返回延迟与吞吐量。
    return latency_ms, throughput


def measure_peak_memory(model: torch.nn.Module, sample: torch.Tensor, device: torch.device):
    """
    测量“单输入模型”的 CUDA 峰值显存占用。

    返回：
        float:
            显存峰值，单位 MB。

    说明：
    - 只有 CUDA 下有意义
    - CPU 模式下直接返回 0.0
    """
    # 如果不是 CUDA，显存统计没有意义，直接返回 0。
    if device.type != "cuda":
        return 0.0

    # 尽量清理缓存，降低历史残留对统计的影响。
    torch.cuda.empty_cache()
    # 重置峰值显存统计器。
    torch.cuda.reset_peak_memory_stats(device)
    # 不需要梯度，只做一次 forward。
    with torch.no_grad():
        _ = model(sample)
        # 等 GPU 真正执行完，峰值显存统计才稳定。
        torch.cuda.synchronize()
    # 返回“已分配显存”的峰值，单位从 Byte 转为 MB。
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def measure_peak_memory_for_inputs(model: torch.nn.Module, inputs: tuple, device: torch.device):
    """
    测量“多输入模型”的 CUDA 峰值显存占用。
    """
    # 非 CUDA 环境直接返回 0。
    if device.type != "cuda":
        return 0.0

    # 清空缓存，减少前序操作污染。
    torch.cuda.empty_cache()
    # 重置峰值显存统计。
    torch.cuda.reset_peak_memory_stats(device)
    # 关闭梯度，只做推理统计。
    with torch.no_grad():
        _ = model(*inputs)
        # 同步 GPU，确保统计完成。
        torch.cuda.synchronize()
    # 从 Byte 转成 MB 返回。
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def build_summary_lines(profile: dict):
    """
    把单个模型的 profiling 结果整理成人类可读的文本行列表。

    参数：
        profile: dict
            包含一个模型 profiling 结果的字典。

    返回：
        list[str]:
            适合直接写入 txt 或打印到终端的多行文本。
    """
    # 先构造通用字段。
    lines = [
        f"Model                  : {profile['model_name']}",
        f"Checkpoint             : {profile['checkpoint']}",
        f"Device                 : {profile['device']}",
        f"Input shape            : {profile['input_shape']}",
        f"Output shape           : {profile['output_shape']}",
        f"Total params           : {profile['total_params']} ({format_count(profile['total_params'])})",
        f"Trainable params       : {profile['trainable_params']} ({format_count(profile['trainable_params'])})",
        f"Estimated FLOPs        : {int(profile['flops'])} ({format_count(profile['flops'])})",
        f"Average latency        : {profile['latency_ms']:.3f} ms / batch",
        f"Throughput             : {profile['throughput']:.3f} images/s",
    ]
    # 如果设备是 CUDA，就额外追加显存信息。
    if profile["device"].startswith("cuda"):
        lines.append(f"Peak GPU memory        : {profile['peak_memory_mb']:.3f} MB")
    else:
        # CPU 模式下明确说明不适用。
        lines.append("Peak GPU memory        : N/A (CPU mode)")
    # 返回整理后的字符串列表。
    return lines


def profile_model(model_name: str, model: torch.nn.Module, sample: torch.Tensor, device: torch.device, checkpoint: str, warmup: int, iters: int):
    """
    对“单输入模型”做一整套 profiling。

    这个函数会统一产出：
    - 参数量
    - FLOPs
    - 延迟
    - 吞吐量
    - 峰值显存
    - 输入/输出形状

    返回：
        dict:
            一份结构化结果，后面可以直接打印或写入 CSV。
    """
    # 把模型移动到目标设备。
    model = model.to(device)
    # 切到 eval，避免 dropout / bn 的训练态影响推理统计。
    model.eval()

    # 统计总参数量与可训练参数量。
    total_params, trainable_params = count_parameters(model)
    # 估算 FLOPs。
    flops = estimate_flops(model, sample, device)
    # 测量延迟和吞吐量。
    latency_ms, throughput = benchmark_latency(model, sample, device, warmup, iters)
    # 测量峰值显存。
    peak_memory_mb = measure_peak_memory(model, sample, device)

    # 再单独跑一次 forward，用来拿输出 shape。
    with torch.no_grad():
        output = model(sample)

    # 返回统一结构的 profiling 结果字典。
    return {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "device": str(device),
        "input_shape": tuple(sample.shape),
        "output_shape": tuple(output.shape),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": float(flops),
        "latency_ms": float(latency_ms),
        "throughput": float(throughput),
        "peak_memory_mb": float(peak_memory_mb),
    }


def save_results(output_dir: Path, tag: str, args, profiles: list[dict]):
    """
    把 profiling 结果同时保存为 txt 和 csv。

    参数：
        output_dir:
            输出目录。
        tag:
            实验标签。
        args:
            parse_args() 返回的命令行参数对象。
        profiles:
            多个模型的 profiling 结果列表。

    返回：
        tuple[Path, Path]:
            - txt_path
            - csv_path
    """
    # 如果目录不存在，就递归创建。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 生成时间戳，用于区分不同运行结果文件。
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 生成 txt 输出路径。
    txt_path = output_dir / f"{tag}_{timestamp}.txt"
    # 生成 csv 输出路径。
    csv_path = output_dir / f"{tag}_{timestamp}.csv"

    # 先整理 txt 的固定头部内容。
    txt_lines = [
        "=== ESRGAN Profiling Summary ===",
        f"Tag                    : {tag}",
        f"Project root           : {args.project_root}",
        f"Warmup                 : {args.warmup}",
        f"Iterations             : {args.iters}",
        "",
    ]
    # 逐个模型追加文本摘要。
    for idx, profile in enumerate(profiles, start=1):
        txt_lines.append(f"[{idx}] {profile['model_name']}")
        txt_lines.extend(build_summary_lines(profile))
        txt_lines.append("")
    # 写入 txt 文件。
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # 统一定义 CSV 表头。
    fieldnames = [
        "tag",
        "model_name",
        "checkpoint",
        "device",
        "batch_size",
        "channels",
        "height",
        "width",
        "input_shape",
        "output_shape",
        "total_params",
        "trainable_params",
        "flops",
        "latency_ms",
        "throughput",
        "peak_memory_mb",
        "warmup",
        "iters",
        "project_root",
        "timestamp",
    ]
    # 以 utf-8-sig 编码打开 CSV，兼容 Excel 打开中文/UTF-8 场景。
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        # 创建 DictWriter，按 fieldnames 顺序写列。
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 写表头。
        writer.writeheader()
        # 逐个模型写一行数据。
        for profile in profiles:
            writer.writerow(
                {
                    "tag": tag,
                    "model_name": profile["model_name"],
                    "checkpoint": profile["checkpoint"],
                    "device": profile["device"],
                    "batch_size": args.batch_size,
                    "channels": args.channels,
                    "height": args.height,
                    "width": args.width,
                    "input_shape": profile["input_shape"],
                    "output_shape": profile["output_shape"],
                    "total_params": profile["total_params"],
                    "trainable_params": profile["trainable_params"],
                    "flops": profile["flops"],
                    "latency_ms": profile["latency_ms"],
                    "throughput": profile["throughput"],
                    "peak_memory_mb": profile["peak_memory_mb"],
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "project_root": str(args.project_root),
                    "timestamp": timestamp,
                }
            )
    # 返回两个输出文件路径，方便上层打印或继续处理。
    return txt_path, csv_path


def main():
    """
    主入口。

    当前版本里只保留了参数解析和设备解析；
    原本 Generator / Discriminator 的 profiling 示例逻辑被注释保留，
    方便你后续按项目里的实际模型结构继续接回去。
    """
    # 读取命令行参数。
    args = parse_args()
    # 解析最终设备。
    device = resolve_device(args.device)
    # 下面原本是 ESRGAN Generator / Discriminator 的 profiling 示例代码。
    # 目前被整体注释掉，说明这个脚本现在主要作为“工具函数库”被别处复用。
    # 如果后续你想恢复独立运行能力，可以把相关模型构建代码接回来。
    # generator = load_generator(args.project_root, args.channels, args.num_blocks, args.scale)
    # discriminator = load_discriminator(args.project_root, args.channels)
    # generator_checkpoint = maybe_load_checkpoint(generator, args.generator_checkpoint, device)
    # discriminator_checkpoint = maybe_load_checkpoint(discriminator, args.discriminator_checkpoint, device)
    #
    # generator_input = torch.randn(
    #     args.batch_size, args.channels, args.height, args.width, device=device
    # )
    # with torch.no_grad():
    #     generator_output = generator.to(device).eval()(generator_input)
    # discriminator_input = generator_output.detach()
    #
    # profiles = [
    #     profile_model(
    #         "Generator",
    #         generator,
    #         generator_input,
    #         device,
    #         generator_checkpoint,
    #         args.warmup,
    #         args.iters,
    #     ),
    #     profile_model(
    #         "Discriminator",
    #         discriminator,
    #         discriminator_input,
    #         device,
    #         discriminator_checkpoint,
    #         args.warmup,
    #         args.iters,
    #     ),
    # ]
    #
    # print("=== ESRGAN Profiling Summary ===")
    # print(f"Project root           : {args.project_root}")
    # print(f"Saved tag              : {args.tag}")
    # print(f"Device                 : {device}")
    # print("")
    # for profile in profiles:
    #     print(f"[{profile['model_name']}]")
    #     for line in build_summary_lines(profile):
    #         print(line)
    #     print("")
    #
    # txt_path, csv_path = save_results(args.output_dir, args.tag, args, profiles)
    # print(f"TXT saved              : {txt_path}")
    # print(f"CSV saved              : {csv_path}")
    #
    # 这里刻意不做额外输出，表示 main 当前只是一个最小入口。
    _ = device


# 只有当这个文件被直接运行时，才进入 main。
# 如果它是被别的模块 import，就不会自动执行。
if __name__ == "__main__":
    main()
