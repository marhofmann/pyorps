"""CLI entry point: ``python -m pyorps.gui [--desktop] [--port N]``."""
from __future__ import annotations

import argparse

from .app import launch


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pyorps.gui",
        description="PYORPS interactive route-planning GUI.")
    parser.add_argument("--desktop", action="store_true",
                        help="open in a desktop window (pywebview)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    launch(host=args.host, port=args.port, debug=args.debug,
           desktop=args.desktop)


if __name__ == "__main__":
    main()
