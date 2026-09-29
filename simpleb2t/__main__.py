"""Run `python -m simpleb2t --help` from this directory."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Local LibriBrain100 experiments: overlap controls and SimpleB2T."
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=Path("runs"),
        help="Local cache, checkpoints, and numerical results (default: runs)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device; cuda for paper-scale runs, cpu for small checks",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download the exact public recording files")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cross-subject", action="store_true")
    p.add_argument("--annotations-only", action="store_true")
    p = sub.add_parser("prepare", help="Filter, scale, and extract all fixed word windows")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cross-subject", action="store_true")
    p.add_argument("--annotations-only", action="store_true")
    p = sub.add_parser("train", help="Train or resume one model and seed")
    p.add_argument(
        "--model",
        choices=[
            "ours",
            "joint",
            "single_word",
            "shared_pulses",
            "independent_pulses",
            "stitched",
            "timing",
        ],
        required=True,
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--full-patience",
        action="store_true",
        help="Ignore historical manual-stop caps; this is not the exact paper schedule",
    )
    p = sub.add_parser("run", help="Run a named paper recipe, skipping completed stages")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument(
        "recipe",
        choices=[
            "all",
            "metadata",
            "train",
            "natural",
            "clinical",
            "ablations",
            "observations",
            "prompts",
            "cross-subject",
            "stitched",
            "timing-agreement",
            "diagnostics",
        ],
    )
    p.add_argument(
        "--published-settings",
        action="store_true",
        help="Reuse published baseline/prompt settings instead of refitting them on development data",
    )
    sub.add_parser("summarize", help="Collect locally computed numeric results")
    sub.add_parser(
        "verify", help="Audit fixed assignments, metadata, and bundle checksums (no GPU)"
    )
    p = sub.add_parser(
        "targets", help="Regenerate frozen T5 targets and compare with the bundled vectors"
    )
    p.add_argument("--output", type=Path, default=Path("regenerated_targets.npz"))
    args = parser.parse_args()
    if args.command in ["download", "prepare"]:
        from . import data

        if args.command == "download":
            data.download(args.data_root, args.cross_subject, args.annotations_only)
        else:
            data.prepare(args.data_root, args.work, args.cross_subject, args.annotations_only)
    elif args.command == "train":
        from .training import run

        run(args.work, args.model, args.seed, args.device, not args.full_patience)
    elif args.command in ["run", "summarize"]:
        from .experiments import run, aggregate

        if args.command == "run":
            run(args.work, args.recipe, args.device, args.published_settings, args.seeds)
        else:
            aggregate(args.work)
    elif args.command == "targets":
        from .verify import regenerate_targets

        regenerate_targets(args.output, args.device)
    else:
        from .verify import verify

        verify()


if __name__ == "__main__":
    main()
