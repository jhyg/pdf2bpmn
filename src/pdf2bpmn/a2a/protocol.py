"""A2A protocol message types and schemas."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskEventType(str, Enum):
    """Event types for task execution."""
    PROGRESS = "progress"
    ARTIFACT = "artifact"
    COMPLETE = "complete"
    ERROR = "error"


class DiscoverResponse(BaseModel):
    """Agent discovery response."""
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    input_schema: Dict[str, Any] = Field(..., description="Input schema for task execution")
    output_schema: Dict[str, Any] = Field(..., description="Output schema for task results")
    version: str = Field(default="1.0.0", description="Agent version")


class ExecuteRequest(BaseModel):
    """Task execution request."""
    input: Dict[str, Any] = Field(..., description="Task input data")
    task_id: Optional[str] = Field(None, description="Optional task ID (for idempotency)")


class ExecuteResponse(BaseModel):
    """Task execution response."""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Initial task status")
    message: Optional[str] = Field(None, description="Optional status message")


class TaskEvent(BaseModel):
    """Task execution event."""
    task_id: str = Field(..., description="Task identifier")
    event_type: TaskEventType = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: Optional[str] = Field(None, description="Event timestamp (ISO format)")


class TaskResult(BaseModel):
    """Task execution result."""
    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Final task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Generated artifacts")


# Default schemas for PDF2BPMN agent
DEFAULT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "pdf_url": {
            "type": "string",
            "description": "URL of the PDF file to analyze"
        },
        "pdf_path": {
            "type": "string",
            "description": "Local file path to the PDF file"
        },
        "pdf_file_name": {
            "type": "string",
            "description": "Original PDF file name"
        }
    },
    "required": []
}

DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "bpmn_xml": {
            "type": "string",
            "description": "Generated BPMN XML"
        },
        "process_count": {
            "type": "integer",
            "description": "Number of processes extracted"
        },
        "processes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "bpmn_xml": {"type": "string"}
                }
            }
        }
    }
}
