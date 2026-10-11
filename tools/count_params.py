"""
    python tools/count_params.py projects/configs/sparsedrive_small_stage2.py
    python tools/count_params.py <config> --depth 2
    python tools/count_params.py <config> --ckpt work_dirs/xxx/iter_xxx.pth
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from mmcv import Config
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("config", help="test config file path")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="if given, additionally count params from the checkpoint's state_dict",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="module tree depth for the per-module breakdown",
    )
    return parser.parse_args()


def load_plugin(cfg, config_path):

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if not getattr(cfg, "plugin", False):
        return
    plugin_dir = getattr(cfg, "plugin_dir", None)
    if plugin_dir is not None:
        _module_dir = os.path.dirname(plugin_dir)
    else:
        _module_dir = os.path.dirname(config_path)
    _module_path = ".".join(_module_dir.split("/"))
    importlib.import_module(_module_path)


def count(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in module.buffers())
    return total, trainable, buffers


def breakdown(model, depth):

    rows = []

    def walk(module, prefix, level):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            n = sum(p.numel() for p in child.parameters())
            if n == 0:
                continue
            rows.append((full, n, level))
            if level < depth:
                walk(child, full, level + 1)

    walk(model, "", 1)
    return rows


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    load_plugin(cfg, args.config)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.eval()

    total, trainable, buffers = count(model)
    print(f"\nconfig: {args.config}")
    print(f"{'=' * 72}")
    print(f"total params      : {total:>12,}  ({total / 1e6:8.2f} M)")
    print(f"trainable params  : {trainable:>12,}  ({trainable / 1e6:8.2f} M)")
    print(f"frozen params     : {total - trainable:>12,}  ({(total - trainable) / 1e6:8.2f} M)")
    print(f"buffers (non-param): {buffers:>11,}  ({buffers / 1e6:8.2f} M)")
    print(f"fp32 size         : {total * 4 / 1024 ** 2:8.2f} MiB")
    print(f"{'=' * 72}")

    print(f"\nper-module breakdown (depth={args.depth}):")
    for name, n, level in breakdown(model, args.depth):
        indent = "  " * (level - 1)
        print(f"  {indent}{name:<48s} {n / 1e6:8.3f} M  ({100 * n / total:5.1f}%)")

    if args.ckpt is not None:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        n_ckpt = sum(v.numel() for v in sd.values() if torch.is_floating_point(v))
        print(f"\ncheckpoint: {args.ckpt}")
        print(f"  state_dict float entries: {n_ckpt / 1e6:.2f} M "
              f"(include: BN running stats, nomally > total params)")


if __name__ == "__main__":
    main()
