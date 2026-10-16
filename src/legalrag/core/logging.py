"""Structured JSON logging for Legal RAG."""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
thread_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("thread_id", default=None)


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        req_id = request_id_ctx.get()
        if req_id:
            log_obj["request_id"] = req_id

        th_id = thread_id_ctx.get()
        if th_id:
            log_obj["thread_id"] = th_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include extra fields
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message"
            } and not key.startswith("_"):
                log_obj[key] = val

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSONFormatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with specified name."""
    return logging.getLogger(name)
