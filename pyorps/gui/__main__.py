"""CLI entry point: ``python -m pyorps.gui [--desktop] [--port N] ...`` (``--help`` lists all options)."""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from .app import LaunchError, check_host, launch
from .callbacks.io import apply_opened_project
from .services import project_io
from .services.errors import ProjectLayersSkippedWarning, ProjectLoadError
from .state import ProjectState, TileManager


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


def _project_path(value: str) -> Path:
    """``--project``: the path made absolute (a relative one is taken from the
    current folder). Whether it exists is not checked here: a missing file ends
    the program with exit code 1, like a broken one, not with a usage error."""
    return Path(os.path.abspath(os.path.expanduser(value)))


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
    parser.add_argument("--project", type=_project_path, default=None,
                        metavar="FILE",
                        help="open a saved project at start: a project.json or "
                             "a folder that holds one (a relative path is taken "
                             "from the current folder). A missing or broken file "
                             "prints one error line and ends the program with "
                             "exit code 1 before the server starts or a window "
                             "opens")
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


def _cannot_open(path: Path, reason: str) -> LaunchError:
    """The one error line of a project that cannot be opened at start."""
    return LaunchError(f"could not open the project '{path}': {reason}")


def _project_manifest(path: Path) -> Path:
    """The ``project.json`` behind ``--project`` (a folder holds one), checked
    BEFORE any state exists; a :class:`LaunchError` when there is none."""
    if path.is_dir():
        manifest = path / "project.json"
        if not manifest.is_file():
            raise _cannot_open(manifest, "the folder holds no project.json")
        return manifest
    if not path.exists():
        raise _cannot_open(path, "the file does not exist")
    if not path.is_file():
        raise _cannot_open(path, "it is not a file")
    return path


def _reason(exc: Exception) -> str:
    """Why a project failed to load, on one line."""
    text = " ".join(str(exc).split())
    if isinstance(exc, ProjectLoadError):          # names the layer and its file
        return text
    if isinstance(exc, OSError):
        return f"the file could not be read ({type(exc).__name__}: {text})"
    return f"the file is not a valid project ({type(exc).__name__}: {text})"


def _discard(state: ProjectState, *, remove_work_dir: bool) -> None:
    """Undo a failed start: stop the tile servers the load may have started and
    remove the temporary work folder when it is still empty (a folder the user
    named with ``--work-dir``, or one that holds files, stays)."""
    for cleanup in (state.shutdown, TileManager.stop_server):
        with contextlib.suppress(Exception):
            cleanup()
    if remove_work_dir:
        with contextlib.suppress(OSError):
            state.work_dir.rmdir()


def open_project_at_start(path: Path, work_dir: Path | None = None) -> ProjectState:
    """A new :class:`ProjectState` holding the project at ``path`` (a
    ``project.json`` or a folder with one), for ``--project``.

    Opened through ``project_io.load_project`` and ``apply_opened_project``,
    exactly like the "Open project" button, so both end in the same state.
    Layers that could not be restored are printed as ``warning: ...`` lines.
    A missing or broken file raises a :class:`LaunchError` with one line that
    names the file; nothing has been served or opened at that point.
    """
    manifest_file = _project_manifest(path)
    state = ProjectState(work_dir=work_dir)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manifest = project_io.load_project(state, manifest_file)
        apply_opened_project(state, manifest, manifest_file)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # any broken file is ONE error line, never a traceback
        _discard(state, remove_work_dir=work_dir is None)
        raise _cannot_open(manifest_file, _reason(exc)) from exc
    for item in caught:
        if issubclass(item.category, ProjectLayersSkippedWarning):
            print(f"warning: {item.message}", file=sys.stderr)
        else:                                   # not ours: let it through
            warnings.warn_explicit(item.message, item.category, item.filename,
                                   item.lineno, source=item.source)
    count = len(state.layers)
    print(f"opened project: {manifest_file} "
          f"({count} layer{'' if count == 1 else 's'})")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.project is not None:
            state = open_project_at_start(args.project, args.work_dir)
        else:
            state = ProjectState(work_dir=args.work_dir) if args.work_dir else None
        launch(state, host=args.host, port=args.port, debug=args.debug,
               desktop=args.desktop, allow_remote=args.allow_remote,
               threads=args.threads, compress=args.compress)
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
