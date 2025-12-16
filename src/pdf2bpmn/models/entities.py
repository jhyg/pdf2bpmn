"""Entity models for the knowledge graph."""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4


def generate_id() -> str:
    return str(uuid4())


class TaskType(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


class AmbiguityStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class ConflictSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FragmentType(str, Enum):
    OVERVIEW = "overview"
    DETAIL = "detail"
    EXCEPTION = "exception"
    NOTE = "note"


class GatewayType(str, Enum):
    EXCLUSIVE = "exclusive"
    PARALLEL = "parallel"
    INCLUSIVE = "inclusive"


class EventType(str, Enum):
    START = "start"
    END = "end"
    INTERMEDIATE = "intermediate"


# Base Entity
class BaseEntity(BaseModel):
    id: str = Field(default_factory=generate_id)
    created_at: datetime = Field(default_factory=datetime.now)
    version: int = 1
    created_by: str = "agent"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


# Document Structure
class Document(BaseEntity):
    doc_id: str = Field(default_factory=generate_id)
    title: str
    source: str  # file path
    uploaded_at: datetime = Field(default_factory=datetime.now)
    page_count: int = 0


class Section(BaseEntity):
    section_id: str = Field(default_factory=generate_id)
    doc_id: str
    heading: str
    level: int
    page_from: int
    page_to: int
    content: str = ""


class ReferenceChunk(BaseEntity):
    chunk_id: str = Field(default_factory=generate_id)
    doc_id: str
    page: int
    span: str  # start:end character positions
    text: str
    hash: str = ""
    embedding: Optional[list[float]] = None


class Evidence(BaseEntity):
    evi_id: str = Field(default_factory=generate_id)
    doc_id: str
    page: int
    text_span: str
    locator: str = ""  # Optional bbox or selector


# Process Hierarchy
class ProcessDefFragment(BaseEntity):
    fragment_id: str = Field(default_factory=generate_id)
    process_id: str
    fragment_type: FragmentType
    text: str
    confidence: float = 1.0


class Process(BaseEntity):
    proc_id: str = Field(default_factory=generate_id)
    name: str
    purpose: str = ""
    triggers: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    description: str = ""


class Task(BaseEntity):
    task_id: str = Field(default_factory=generate_id)
    process_id: str = ""
    name: str
    task_type: TaskType = TaskType.HUMAN
    description: str = ""
    order: int = 0


class Role(BaseEntity):
    role_id: str = Field(default_factory=generate_id)
    name: str
    org_unit: str = ""
    persona_hint: str = ""


class Gateway(BaseEntity):
    gateway_id: str = Field(default_factory=generate_id)
    process_id: str = ""
    name: str = ""
    gateway_type: GatewayType = GatewayType.EXCLUSIVE
    condition: str = ""
    description: str = ""


class Event(BaseEntity):
    event_id: str = Field(default_factory=generate_id)
    process_id: str = ""
    event_type: EventType
    name: str
    trigger: str = ""


# Skill for Agent Tasks
class Skill(BaseEntity):
    skill_id: str = Field(default_factory=generate_id)
    name: str
    summary: str = ""
    purpose: str = ""
    inputs: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    md_path: str = ""


# DMN (Decision Model and Notation)
class DMNDecision(BaseEntity):
    decision_id: str = Field(default_factory=generate_id)
    name: str
    description: str = ""
    input_data: list[str] = Field(default_factory=list)
    output_data: list[str] = Field(default_factory=list)


class DMNRule(BaseEntity):
    rule_id: str = Field(default_factory=generate_id)
    decision_id: str
    when: str  # condition
    then: str  # result
    confidence: float = 1.0


# HITL and Quality
class Ambiguity(BaseEntity):
    amb_id: str = Field(default_factory=generate_id)
    entity_type: str  # Process, Task, Role, Gateway, DMNDecision
    entity_id: str
    question: str
    options: list[str] = Field(default_factory=list)
    status: AmbiguityStatus = AmbiguityStatus.OPEN
    answer: Optional[str] = None


class Alias(BaseEntity):
    alias_id: str = Field(default_factory=generate_id)
    entity_type: str
    entity_id: str
    text: str
    normalized: str = ""


class Conflict(BaseEntity):
    conflict_id: str = Field(default_factory=generate_id)
    conflict_type: str
    description: str
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    status: AmbiguityStatus = AmbiguityStatus.OPEN
    entities: list[str] = Field(default_factory=list)




