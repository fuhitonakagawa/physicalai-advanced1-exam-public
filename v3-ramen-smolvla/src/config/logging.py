"""相関IDを持つJSON Lines形式のアプリログと監査ログ。"""

import logging
import sys
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

ALLOWED_FIELDS = frozenset(
    {
        "event",
        "level",
        "logger",
        "timestamp",
        "correlation_id",
        "mode",
        "action_count",
        "success_count",
        "error_type",
        "exception",
        "episode",
        "success",
        "steps",
    }
)
ALLOWED_EVENTS = frozenset(
    {
        "run_started",
        "plan_accepted",
        "plan_rejected",
        "run_finished",
        "run_failed",
        "configuration_invalid",
        "model_loading",
        "model_ready",
        "episode_finished",
        "episode_progress",
        "smoke_finished",
    }
)


def timestamp(
    logger: object, method_name: str, event_dict: MutableMapping[str, Any]
) -> dict[str, Any]:
    """JSTの時刻を付与する。"""
    event_dict["timestamp"] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    return dict(event_dict)


def redact(
    logger: object, method_name: str, event_dict: MutableMapping[str, Any]
) -> dict[str, Any]:
    """許可されたフィールドと固定イベントだけをログに含める。"""
    result = {key: value for key, value in event_dict.items() if key in ALLOWED_FIELDS}
    if result.get("event") not in ALLOWED_EVENTS:
        result["event"] = "external_event"
    result.setdefault("correlation_id", "unbound")
    if "exception" in result:
        result["exception"] = [
            {
                "exc_type": item.get("exc_type"),
                "frames": [
                    {"name": frame.get("name"), "lineno": frame.get("lineno")}
                    for frame in item.get("frames", [])
                ],
            }
            for item in result["exception"]
        ]
    return result


def configure_logging(level: str) -> None:
    """アプリログを標準出力、監査ログを標準エラーへ振り分ける。"""
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamp,
        structlog.processors.dict_tracebacks,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            redact,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    audit = logging.getLogger("audit")
    audit.handlers.clear()
    audit.propagate = False
    audit.setLevel(level)
    audit_handler = logging.StreamHandler(sys.stderr)
    audit_handler.setFormatter(formatter)
    audit.addHandler(audit_handler)
    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
