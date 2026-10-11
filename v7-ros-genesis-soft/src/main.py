"""実行設定に対応するモデル検査または物理実験。"""

import logging
import sys
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402
from src.config.settings import load_settings  # noqa: E402


def main() -> int:
    """設定と実行のエラーを終了コードへ変換する。"""
    try:
        settings = load_settings()
    except (ValueError, OSError):
        configure_logging("ERROR")
        structlog.contextvars.bind_contextvars(correlation_id=uuid4().hex)
        structlog.wrap_logger(logging.getLogger(__name__)).exception(
            "configuration_invalid"
        )
        return 1
    configure_logging(settings.log_level)
    from src.pipeline import run

    try:
        run(settings)
    except Exception as error:
        structlog.wrap_logger(logging.getLogger(__name__)).exception(
            "run_failed", error_type=type(error).__name__
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
