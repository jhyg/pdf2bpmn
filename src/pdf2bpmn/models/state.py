"""LangGraph state definition."""
from typing import Annotated, TypedDict
from operator import add

from .entities import (
    Document,
    Section,
    Process,
    Task,
    Role,
    Gateway,
    Event,
    Skill,
    DMNDecision,
    DMNRule,
    Evidence,
    Ambiguity,
    ReferenceChunk,
)


def merge_list(existing: list, new: list) -> list:
    """Merge two lists, avoiding duplicates by id."""
    existing_ids = {getattr(item, 'id', str(i)) for i, item in enumerate(existing)}
    result = list(existing)
    for item in new:
        item_id = getattr(item, 'id', None)
        if item_id and item_id not in existing_ids:
            result.append(item)
            existing_ids.add(item_id)
        elif item_id is None:
            result.append(item)
    return result


class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    
    # Input
    pdf_paths: list[str]
    
    # Document structure
    documents: Annotated[list[Document], merge_list]
    sections: Annotated[list[Section], merge_list]
    reference_chunks: Annotated[list[ReferenceChunk], merge_list]
    
    # Extracted entities
    processes: Annotated[list[Process], merge_list]
    tasks: Annotated[list[Task], merge_list]
    roles: Annotated[list[Role], merge_list]
    gateways: Annotated[list[Gateway], merge_list]
    events: Annotated[list[Event], merge_list]
    
    # Generated artifacts
    skills: Annotated[list[Skill], merge_list]
    dmn_decisions: Annotated[list[DMNDecision], merge_list]
    dmn_rules: Annotated[list[DMNRule], merge_list]
    
    # Evidence and tracking
    evidences: Annotated[list[Evidence], merge_list]
    
    # HITL
    open_questions: Annotated[list[Ambiguity], merge_list]
    resolved_questions: Annotated[list[Ambiguity], merge_list]
    current_question: Ambiguity | None
    user_answer: str | None
    
    # Control flow
    confidence_threshold: float
    current_step: str
    error: str | None
    
    # Outputs
    bpmn_xml: str | None  # Main BPMN XML (backward compatibility - first process)
    bpmn_xmls: dict[str, str]  # process_id -> bpmn_xml (all processes)
    bpmn_files: dict[str, str]  # process_id -> file_path (all BPMN files)
    skill_docs: dict[str, str]  # skill_id -> markdown content
    dmn_xml: str | None




