"""
PYORPS GUI: persistent notice log.

Every notification the GUI shows (errors, warnings, info, success) is also
written to a rotating log file the user can read at any time — for later
development, testing and bug fixing. The file lives in a STABLE location
(``~/.pyorps/logs/pyorps_gui.log``, overridable via ``PYORPS_LOG_DIR``) that
survives a session (unlike the temp work dir), and is rotated daily, keeping a
few days of history.

The GUI reads it back through :func:`read_log` (the "View log" panel); the
notification callback funnels every new notice through :func:`log_notice`.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

#: keep this many past days of rotated logs (plus today's file)
BACKUP_DAYS = 5
LOG_FILENAME = "pyorps_gui.log"

_SEVERITY_LEVEL = {"error": logging.ERROR, "warning": logging.WARNING,
                   "info": logging.INFO, "success": logging.INFO}

_logger: logging.Logger | None = None
_log_dir: Path | None = None


def _resolve_dir() -> Path:
    override = os.environ.get("PYORPS_LOG_DIR")
    return Path(override) if override else (Path.home() / ".pyorps" / "logs")


def log_file_path() -> Path:
    """Absolute path of the current log file (its directory is created lazily)."""
    return (_log_dir or _resolve_dir()) / LOG_FILENAME


def get_logger() -> logging.Logger:
    """The shared, file-backed logger (created once per process)."""
    global _logger, _log_dir  # pylint: disable=global-statement  # module-level cached logger
    if _logger is not None:
        return _logger
    _log_dir = _resolve_dir()
    _log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pyorps.gui.logbook")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.handlers.TimedRotatingFileHandler(
            str(_log_dir / LOG_FILENAME), when="midnight",
            backupCount=BACKUP_DAYS, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    _logger = logger
    return logger


def log_notice(notice: dict) -> None:
    """Append one notice (services.errors.Notice.to_dict) to the log file."""
    severity = notice.get("severity", "info")
    level = _SEVERITY_LEVEL.get(severity, logging.INFO)
    parts = [f"[{severity}] {notice.get('title', 'Notice')}"]
    for label, key in (("Means", "meaning"), ("Impact", "impact"),
                       ("Fix", "fix")):
        text = notice.get(key)
        if text:
            parts.append(f"{label}: {text}")
    logger = get_logger()
    logger.log(level, " | ".join(parts))
    details = notice.get("details")
    if details and level >= logging.WARNING:
        indented = str(details).rstrip().replace("\n", "\n    ")
        logger.log(level, "    %s", indented)


def read_log(max_lines: int = 800) -> str:
    """The tail (last ``max_lines`` lines) of the current log file, as text."""
    try:
        text = log_file_path().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "(no log entries yet — warnings and errors will appear here)"
    lines = text.splitlines()
    tail = lines[-max_lines:]
    if len(lines) > max_lines:
        tail = [f"… ({len(lines) - max_lines} earlier line(s) omitted)", *tail]
    return "\n".join(tail) or "(log file is empty)"


def reset() -> None:
    """Drop the cached logger + handlers (tests, to redirect the log dir)."""
    global _logger, _log_dir  # pylint: disable=global-statement  # module-level cached logger
    if _logger is not None:
        for handler in list(_logger.handlers):
            handler.close()
            _logger.removeHandler(handler)
    _logger = None
    _log_dir = None
