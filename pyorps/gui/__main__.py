"""CLI entry point: ``python -m pyorps.gui [--desktop] [--port N] ...`` (``--help`` lists all options)."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .app import LaunchError, check_host, launch
from .state import ProjectState


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _folder(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"{value!r} exists and is not a folder")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyorps.gui",
        description="PYORPS interactive route-planning GUI. It starts a local "
                    "web server and prints its address; with --desktop it opens "
                    "a desktop window instead.")
    parser.add_argument("--desktop", action="store_true",
                        help="open a desktop window (pywebview) instead of "
                             "serving to a browser tab")
    parser.add_argument("--host", default="127.0.0.1", metavar="HOST",
                        help="address to listen on; anything but 127.0.0.1, "
                             "localhost or ::1 needs --allow-remote "
                             "(default: %(default)s)")
    parser.add_argument("--port", type=_port, default=8050, metavar="PORT",
                        help="TCP port; if it is busy the next free one is used "
                             "(default: %(default)s)")
    parser.add_argument("--allow-remote", action="store_true",
                        help="allow a non-loopback --host; the GUI has NO "
                             "authentication and its path fields reach every "
                             "file of this user, so use trusted networks only")
    parser.add_argument("--threads", type=_positive, default=8, metavar="N",
                        help="worker threads of the web server "
                             "(default: %(default)s)")
    parser.add_argument("--work-dir", type=_folder, default=None, metavar="DIR",
                        help="folder for the session's rasters and temporary "
                             "files (default: a new pyorps_gui_* folder in the "
                             "temp directory; never deleted automatically)")
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="compress HTTP responses (default: on in the "
                             "browser when flask-compress is installed, off "
                             "with --desktop)")
    parser.add_argument("--debug", action="store_true",
                        help="Flask/Dash development server with debug tools "
                             "and auto-reload (slower; for development only)")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; a refused host ends in a usage error (exit 2)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        check_host(args.host, args.allow_remote)
    except LaunchError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = ProjectState(work_dir=args.work_dir) if args.work_dir else None
    try:
        launch(state, host=args.host, port=args.port, debug=args.debug,
               desktop=args.desktop, allow_remote=args.allow_remote,
               threads=args.threads, compress=args.compress)
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
