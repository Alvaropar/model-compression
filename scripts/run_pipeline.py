"""
Entry-point script for the Qwen3-VL model compression pipeline.

Usage examples
--------------
# Full pipeline
python scripts/run_pipeline.py --config configs/compression_config.yaml

# Skip recovery (profiling + pruning only)
python scripts/run_pipeline.py --config configs/compression_config.yaml --skip recovery

# Only run SVD profiling
python scripts/run_pipeline.py --config configs/compression_config.yaml --skip width depth recovery

# Resume after width pruning (skip SVD and width, run depth + recovery)
python scripts/run_pipeline.py --config configs/compression_config.yaml --skip svd width

# Override device
python scripts/run_pipeline.py --config configs/compression_config.yaml --device cuda:1
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.pipeline import CompressionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Model Compression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/compression_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["svd", "width", "depth", "recovery"],
        default=[],
        help="Phases to skip. Can specify multiple: --skip svd width",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. cuda:0, cpu). Default: auto.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run evaluation (perplexity) after compression.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level),
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    device = torch.device(args.device) if args.device else None

    pipeline = CompressionPipeline.from_config(
        config_path=args.config,
        skip_phases=set(args.skip),
        device=device,
        run_eval=args.eval,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
