"""Explicit real-data stage commands; full experiments run only when requested."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def _json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "frequency":
        from .frequency import main as frequency_main
        return frequency_main(argv[1:])
    parser = argparse.ArgumentParser(prog="python -m microscan")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Real staged training; never launches simulations")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, help="Bounded optimizer check, with resumable interruption")
    for name in ("infer", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--split", default="validation")
        p.add_argument("--device", default="cpu")
        p.add_argument("--phi-config", help="Required explicitly for experimental C only")
        if name == "infer": p.add_argument("--output", required=True)
    p = commands.add_parser("convert", help="Convert supplied calibrated sphere pilot into scalable real-data manifest")
    p.add_argument("--mat", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--grid-size", type=int, default=128)
    p = commands.add_parser("validate")
    p.add_argument("--manifest", required=True)
    p = commands.add_parser("qualify", help="Independent held-out Phi validation and eligibility report")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p = commands.add_parser("profile", help="Bounded structural latency/memory/AMP check; no accuracy claim")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--output")
    args = parser.parse_args(argv)
    from . import training
    if args.command == "train": result = training.train(_json(args.config), resume=args.resume, max_steps=args.max_steps)
    elif args.command in ("infer", "evaluate"):
        kwargs = dict(checkpoint=args.checkpoint, manifest=args.manifest, split=args.split, device=args.device,
                      phi_config=_json(args.phi_config) if args.phi_config else None)
        if args.command == "infer": kwargs["output"] = args.output
        result = getattr(training, args.command)(**kwargs)
    elif args.command == "convert":
        from .data import convert_pilot
        result = {"manifest": str(convert_pilot(args.mat, args.output, args.grid_size))}
    elif args.command == "validate":
        from .data import ManifestDataset, validate_manifest
        manifest = _json(args.manifest)
        result = validate_manifest(manifest)
        # Validate actual arrays too; sealed tests are never loaded by this command.
        for split in ("train", "validation"):
            for sample in ManifestDataset(args.manifest, split):
                pass
        result["arrays_checked"] = ["train", "validation"]
        result["test_arrays_accessed"] = False
    elif args.command == "qualify":
        result = training.qualify(args.checkpoint, args.manifest, _json(args.config), output=args.output, device=args.device)
    elif args.command == "profile":
        if args.steps < 1: parser.error("--steps must be positive")
        result = training.profile(_json(args.config), device=args.device, steps=args.steps, backward=args.backward)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2, default=str))
    return result
