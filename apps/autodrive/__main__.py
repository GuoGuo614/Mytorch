"""Unified CLI: python -m apps.autodrive COMMAND [options]."""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "audit", "manifest", "train",
                                            "evaluate", "drive", "gradcam"))
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    importlib.import_module(f"apps.autodrive.{args.command}").main()


if __name__ == "__main__":
    main()
