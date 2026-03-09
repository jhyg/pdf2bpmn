"""A2A (Agent-to-Agent) protocol implementation for PDF2BPMN."""

from .protocol import (
    DiscoverResponse,
    ExecuteRequest,
    ExecuteResponse,
    TaskStatus,
    TaskEvent,
    TaskResult,
)
from .server import A2AServer
from .client import A2AClient

__all__ = [
    "DiscoverResponse",
    "ExecuteRequest",
    "ExecuteResponse",
    "TaskStatus",
    "TaskEvent",
    "TaskResult",
    "A2AServer",
    "A2AClient",
]
