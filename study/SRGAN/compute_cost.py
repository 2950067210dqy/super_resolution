import argparse
import csv
import sys
import time
from pathlib import Path

import torch


DEFAULT_PROJECT_ROOT = Path(r"D:\WorkSpace\super_reloution_project\study")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile ESRGAN generator/discriminator: params, FLOPs, latency, throughput, and GPU memory."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root that contains the SRGAN package.",
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=None,
        help="Optional generator checkpoint (.pth) to load before profiling.",
    )
    parser.add_argument(
        "--discriminator-checkpoint",
        type=Path,
        default=None,
        help="Optional discriminator checkpoint (.pth) to load before profiling.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--channels", type=int, default=3, help="Input image channels.")
    parser.add_argument("--height", type=int, default=64, help="LR input height.")
    parser.add_argument("--width", type=int, default=64, help="LR input width.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=23,
        help="Number of RRDB blocks used by Generator.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Internal upsample factor used by each upsample stage in Generator.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run profiling on.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "profile_results",
        help="Directory used to save txt/csv profiling results.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="esrgan_profile",
        help="Experiment tag used in saved filenames and CSV rows.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --device cuda was requested.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(project_root: Path, channels: int, num_blocks: int, scale: int):
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    sys.path.insert(0, str(project_root))
    from SRGAN.model.esrgan_update.Module.model import Generator

    return Generator(inner_chanel=channels, num_residual_blocks=num_blocks, scale=scale)


def load_discriminator(project_root: Path, channels: int):
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from SRGAN.model.esrgan_update.Module.model import Discriminator

    return Discriminator(inner_chanel=channels)


def maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    if checkpoint_path is None:
        return "None"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return str(checkpoint_path)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_count(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return f"{value:.0f}"


def estimate_flops(model: torch.nn.Module, sample: torch.Tensor, device: torch.device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            _ = model(sample)

    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops is not None:
            total_flops += event_flops
    return total_flops


def benchmark_latency(model: torch.nn.Module, sample: torch.Tensor, device: torch.device, warmup: int, iters: int):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / iters
    throughput = iters * sample.shape[0] / elapsed
    return latency_ms, throughput


def measure_peak_memory(model: torch.nn.Module, sample: torch.Tensor, device: torch.device):
    if device.type != "cuda":
        return 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(sample)
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def build_summary_lines(profile: dict):
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
    if profile["device"].startswith("cuda"):
        lines.append(f"Peak GPU memory        : {profile['peak_memory_mb']:.3f} MB")
    else:
        lines.append("Peak GPU memory        : N/A (CPU mode)")
    return lines


def profile_model(model_name: str, model: torch.nn.Module, sample: torch.Tensor, device: torch.device, checkpoint: str, warmup: int, iters: int):
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    flops = estimate_flops(model, sample, device)
    latency_ms, throughput = benchmark_latency(model, sample, device, warmup, iters)
    peak_memory_mb = measure_peak_memory(model, sample, device)

    with torch.no_grad():
        output = model(sample)

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
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    txt_path = output_dir / f"{tag}_{timestamp}.txt"
    csv_path = output_dir / f"{tag}_{timestamp}.csv"

    txt_lines = [
        "=== ESRGAN Profiling Summary ===",
        f"Tag                    : {tag}",
        f"Project root           : {args.project_root}",
        f"Warmup                 : {args.warmup}",
        f"Iterations             : {args.iters}",
        "",
    ]
    for idx, profile in enumerate(profiles, start=1):
        txt_lines.append(f"[{idx}] {profile['model_name']}")
        txt_lines.extend(build_summary_lines(profile))
        txt_lines.append("")
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

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
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
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
    return txt_path, csv_path


def main():
    args = parse_args()
    device = resolve_device(args.device)
    generator = load_generator(args.project_root, args.channels, args.num_blocks, args.scale)
    discriminator = load_discriminator(args.project_root, args.channels)
    generator_checkpoint = maybe_load_checkpoint(generator, args.generator_checkpoint, device)
    discriminator_checkpoint = maybe_load_checkpoint(discriminator, args.discriminator_checkpoint, device)

    generator_input = torch.randn(
        args.batch_size, args.channels, args.height, args.width, device=device
    )
    with torch.no_grad():
        generator_output = generator.to(device).eval()(generator_input)
    discriminator_input = generator_output.detach()

    profiles = [
        profile_model(
            "Generator",
            generator,
            generator_input,
            device,
            generator_checkpoint,
            args.warmup,
            args.iters,
        ),
        profile_model(
            "Discriminator",
            discriminator,
            discriminator_input,
            device,
            discriminator_checkpoint,
            args.warmup,
            args.iters,
        ),
    ]

    print("=== ESRGAN Profiling Summary ===")
    print(f"Project root           : {args.project_root}")
    print(f"Saved tag              : {args.tag}")
    print(f"Device                 : {device}")
    print("")
    for profile in profiles:
        print(f"[{profile['model_name']}]")
        for line in build_summary_lines(profile):
            print(line)
        print("")

    txt_path, csv_path = save_results(args.output_dir, args.tag, args, profiles)
    print(f"TXT saved              : {txt_path}")
    print(f"CSV saved              : {csv_path}")


if __name__ == "__main__":
    main()
