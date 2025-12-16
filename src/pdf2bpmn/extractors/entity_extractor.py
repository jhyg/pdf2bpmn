"""LLM-based entity extraction from text."""
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ..config import Config
from ..models.entities import (
    Process, Task, Role, Gateway, Event,
    DMNDecision, DMNRule, TaskType, GatewayType, EventType,
    generate_id
)


class ExtractedEntities(BaseModel):
    """Extracted entities from text."""
    processes: list[dict] = Field(default_factory=list)
    tasks: list[dict] = Field(default_factory=list)
    roles: list[dict] = Field(default_factory=list)
    gateways: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    rules: list[dict] = Field(default_factory=list)
    # Relationships
    task_role_mappings: list[dict] = Field(default_factory=list)
    task_process_mappings: list[dict] = Field(default_factory=list)
    # Sequence flows between tasks
    sequence_flows: list[dict] = Field(default_factory=list)


EXTRACTION_PROMPT = """You are an expert at extracting business process elements from Korean business documents.
{existing_context}
Analyze the following text and extract:
1. **Processes**: Business processes or procedures (절차, 업무 흐름, 처리 단계, 프로세스)
2. **Tasks/Activities**: Individual activities or actions (행위: ~한다, ~해야 한다, 점검, 승인, 검토, 접수, 등록, 통보, 보고)
3. **Roles**: Actors or performers (담당자, 승인권자, 검토자, 부서명, 직책, 시스템, 외부기관)
4. **Gateways**: Decision points or conditions (~인 경우, 아닌 경우, 다만, 예외적으로, 필요 시)
5. **Events**: Start/End triggers (요청 접수 시, 신청서 제출 후, 정기적으로, 완료 시)
6. **Decisions**: Business rules or decision logic (if-then rules, conditions)
7. **Rules**: Specific decision rules (조건-결과 pairs)

IMPORTANT: Also extract RELATIONSHIPS between entities:
8. **task_role_mappings**: Which role performs which task
9. **task_process_mappings**: Which process contains which task
10. **sequence_flows**: The order/sequence between tasks (VERY IMPORTANT!)
    - Identify which task comes BEFORE and AFTER another
    - Look for sequential keywords: "다음", "이후", "후에", "그 다음", "완료 후", "then", "next", "after"
    - Look for numbered steps: 1단계, 2단계, Step 1, Step 2
    - If tasks appear in a numbered list, they are sequential

For each entity, provide:
- name: Clear, concise name
- description: Brief description from the text
- evidence: The exact text span that supports this extraction
- confidence: Your confidence level (0.0 to 1.0)
- order: Sequential order number if identifiable (1, 2, 3...)

For tasks, also identify:
- task_type: "human" (사람이 수행), "agent" (AI/자동화 가능), "system" (시스템 자동)
- performer_role: Name of the role that performs this task (IMPORTANT!)
- parent_process: Name of the process this task belongs to (IMPORTANT!)
- order: Sequential order within the process (1, 2, 3...)
- next_task: Name of the task that follows this one (if identifiable)
- previous_task: Name of the task that precedes this one (if identifiable)

For gateways:
- gateway_type: "exclusive" (XOR), "parallel" (AND), "inclusive" (OR)
- condition: The condition being evaluated
- parent_process: Name of the process this gateway belongs to
- incoming_task: Name of the task before this gateway
- outgoing_tasks: List of task names after this gateway with their conditions

For decisions:
- input_data: List of input data items
- output_data: List of output data items
- related_role: Role that makes this decision

For task_role_mappings:
- task_name: Name of the task
- role_name: Name of the role that performs it

For task_process_mappings:
- task_name: Name of the task
- process_name: Name of the parent process

For sequence_flows (IMPORTANT - extract the order of tasks!):
- from_task: Name of the source task
- to_task: Name of the target task
- condition: Condition for this flow (if any, e.g., "승인인 경우", "거부인 경우")
- process_name: Name of the process this flow belongs to

TEXT TO ANALYZE:
{text}

Respond with a JSON object containing arrays for each entity type.
Return ONLY valid JSON, no markdown formatting."""


# Context template for existing processes/roles/tasks
EXISTING_CONTEXT_TEMPLATE = """
**IMPORTANT - EXISTING ENTITIES (이미 추출된 엔티티들):**

{process_list}
{role_list}
{task_list}

**CRITICAL RULES FOR PROCESS IDENTIFICATION (프로세스 식별 규칙):**
1. If a task clearly belongs to an EXISTING process listed above, use that EXACT process name for parent_process.
   (태스크가 위에 나열된 기존 프로세스에 속하면, 정확히 그 프로세스 이름을 parent_process로 사용하세요)

2. Do NOT create a new process if the content describes steps/tasks of an existing process.
   (내용이 기존 프로세스의 단계/태스크를 설명하는 경우 새 프로세스를 만들지 마세요)

3. "발주 처리", "입고 검수", "대금 지급" etc. are likely TASKS within a larger process, NOT separate processes.
   ("발주 처리", "입고 검수", "대금 지급" 등은 별도 프로세스가 아니라 상위 프로세스의 태스크일 가능성이 높습니다)

4. Look for phrases like "~단계", "~절차", "제N조" which indicate sub-steps of an existing process.
   ("~단계", "~절차", "제N조" 같은 표현은 기존 프로세스의 하위 단계를 나타냅니다)

5. Only create a NEW process if the text explicitly defines a completely different business process.
   (텍스트가 완전히 다른 업무 프로세스를 명시적으로 정의하는 경우에만 새 프로세스를 생성하세요)

6. For existing roles, use the EXACT same name - do not create duplicates with slightly different names.
   (기존 역할의 경우 정확히 같은 이름을 사용하세요 - 약간 다른 이름으로 중복 생성하지 마세요)

**CRITICAL RULES FOR TASK DEDUPLICATION (태스크 중복 제거 규칙):**
7. BEFORE creating a new task, check the EXISTING TASKS list above. If a similar task already exists, DO NOT create a duplicate.
   (새 태스크를 만들기 전에 위의 기존 태스크 목록을 확인하세요. 유사한 태스크가 이미 있으면 중복 생성하지 마세요)

8. Tasks performed by the SAME ROLE in sequence WITHOUT requiring other department collaboration should be MERGED into ONE task.
   (다른 부서 협업 없이 같은 역할이 연속으로 수행하는 작업은 하나의 태스크로 통합해야 합니다)
   Examples of tasks to MERGE (통합해야 할 태스크 예시):
   - "구매요청서 접수" + "형식 검토" → "구매요청서 접수 및 형식 검토" (same role: 구매담당자)
   - "견적서 검토" + "견적 비교" → "견적서 검토 및 비교" (same role)

9. Tasks are SEPARATE only when:
   - Different roles perform them (역할이 다를 때)
   - There is a waiting/approval point between them (승인/대기 시점이 있을 때)
   - They belong to different stages requiring handoff (업무 인계가 필요한 다른 단계일 때)

10. If the current text provides MORE DETAIL about an existing task, DO NOT create a new task.
    Instead, the existing task's description should be enhanced (but this is handled in post-processing).
    (현재 텍스트가 기존 태스크에 대한 추가 설명이면 새 태스크를 만들지 마세요)

"""


class EntityExtractor:
    """Extract business process entities using LLM."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0
        )
        self.parser = JsonOutputParser(pydantic_object=ExtractedEntities)
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        self.chain = self.prompt | self.llm | self.parser
    
    def _build_context(
        self, 
        existing_processes: list[str] = None, 
        existing_roles: list[str] = None,
        existing_tasks: list[dict] = None
    ) -> str:
        """기존 프로세스/역할/태스크 목록으로 컨텍스트 문자열 생성
        
        Args:
            existing_processes: 기존 프로세스 이름 목록
            existing_roles: 기존 역할 이름 목록
            existing_tasks: 기존 태스크 목록 [{name, role, process, order}, ...]
        """
        if not existing_processes and not existing_roles and not existing_tasks:
            return ""
        
        process_list = ""
        if existing_processes:
            process_list = "**기존 프로세스 목록 (Existing Processes):**\n" + \
                          "\n".join(f"  - {p}" for p in existing_processes)
        
        role_list = ""
        if existing_roles:
            role_list = "**기존 역할 목록 (Existing Roles):**\n" + \
                       "\n".join(f"  - {r}" for r in existing_roles)
        
        task_list = ""
        if existing_tasks:
            task_entries = []
            for t in existing_tasks:
                entry = f"  - {t.get('name', 'Unknown')}"
                if t.get('role'):
                    entry += f" (담당: {t['role']})"
                if t.get('process'):
                    entry += f" [프로세스: {t['process']}]"
                task_entries.append(entry)
            task_list = "**기존 태스크 목록 (Existing Tasks - DO NOT DUPLICATE):**\n" + \
                       "\n".join(task_entries)
        
        return EXISTING_CONTEXT_TEMPLATE.format(
            process_list=process_list,
            role_list=role_list,
            task_list=task_list
        )
    
    def extract_from_text(
        self, 
        text: str,
        existing_processes: list[str] = None,
        existing_roles: list[str] = None,
        existing_tasks: list[dict] = None
    ) -> ExtractedEntities:
        """Extract entities from text using LLM.
        
        Args:
            text: 분석할 텍스트
            existing_processes: 이미 추출된 프로세스 이름 목록
            existing_roles: 이미 추출된 역할 이름 목록
            existing_tasks: 이미 추출된 태스크 목록 [{name, role, process, order}, ...]
        
        Returns:
            ExtractedEntities: 추출된 엔티티들
        """
        try:
            # 기존 컨텍스트 생성 (프로세스, 역할, 태스크 모두 포함)
            existing_context = self._build_context(
                existing_processes, 
                existing_roles,
                existing_tasks
            )
            
            result = self.chain.invoke({
                "text": text,
                "existing_context": existing_context
            })
            return ExtractedEntities(**result)
        except Exception as e:
            print(f"Extraction error: {e}")
            return ExtractedEntities()
    
    def convert_to_entities(
        self, 
        extracted: ExtractedEntities,
        doc_id: str,
        chunk_id: str = "",
        existing_processes: dict = None,
        existing_roles: dict = None
    ) -> dict[str, Any]:
        """Convert extracted data to entity objects with relationships."""
        existing_processes = existing_processes or {}
        existing_roles = existing_roles or {}
        
        entities = {
            "processes": [],
            "tasks": [],
            "roles": [],
            "gateways": [],
            "events": [],
            "decisions": [],
            "rules": [],
            "evidences": [],
            # Relationship mappings
            "task_role_map": {},  # task_id -> role_id
            "task_process_map": {},  # task_id -> process_id
            "role_decision_map": {},  # role_id -> [decision_ids]
            "entity_chunk_map": {},  # entity_id -> chunk_id (for evidence)
            # Sequence flows (task ordering)
            "sequence_flows": [],  # list of {from_task_id, to_task_id, condition}
        }
        
        # Task name to ID mapping (built as we create tasks)
        task_name_to_id = {}
        
        # Create name -> id mappings for linking
        process_name_to_id = dict(existing_processes)
        role_name_to_id = dict(existing_roles)
        
        # Convert processes
        for p in extracted.processes:
            proc_id = generate_id()
            proc = Process(
                proc_id=proc_id,
                name=p.get("name", "Unknown Process"),
                purpose=p.get("description", ""),
                description=p.get("description", "")
            )
            entities["processes"].append(proc)
            process_name_to_id[proc.name.lower()] = proc_id
            
            # Link to source chunk
            if chunk_id:
                entities["entity_chunk_map"][proc_id] = chunk_id
        
        # Convert roles
        for r in extracted.roles:
            role_name = r.get("name", "Unknown Role")
            role_key = role_name.lower()
            
            # Check if role already exists
            if role_key not in role_name_to_id:
                role_id = generate_id()
                role = Role(
                    role_id=role_id,
                    name=role_name,
                    org_unit=r.get("org_unit", ""),
                    persona_hint=r.get("persona_hint", r.get("description", ""))
                )
                entities["roles"].append(role)
                role_name_to_id[role_key] = role_id
                
                if chunk_id:
                    entities["entity_chunk_map"][role_id] = chunk_id
        
        # Convert tasks with relationships
        for i, t in enumerate(extracted.tasks):
            task_type_str = (t.get("task_type") or "human").lower()
            task_type = TaskType.HUMAN
            if task_type_str == "agent":
                task_type = TaskType.AGENT
            elif task_type_str == "system":
                task_type = TaskType.SYSTEM
            
            task_id = generate_id()
            
            # Find parent process
            parent_process_name = (t.get("parent_process") or "").lower()
            process_id = ""
            if parent_process_name:
                process_id = process_name_to_id.get(parent_process_name, "")
            # If not found, use first extracted process
            if not process_id and entities["processes"]:
                process_id = entities["processes"][0].proc_id
            
            # Get order from extracted data or use index
            task_order = t.get("order")
            if task_order is None:
                task_order = i
            elif isinstance(task_order, (int, float)):
                task_order = int(task_order)
            elif isinstance(task_order, str):
                try:
                    task_order = int(float(task_order))
                except:
                    task_order = i
            else:
                task_order = i
            
            task_name = t.get("name", f"Task {i+1}")
            
            task = Task(
                task_id=task_id,
                process_id=process_id,
                name=task_name,
                task_type=task_type,
                description=t.get("description", ""),
                order=task_order
            )
            entities["tasks"].append(task)
            
            # Store task name -> id mapping for sequence flow resolution
            task_name_to_id[task_name.lower()] = task_id
            
            # Map task to process
            if process_id:
                entities["task_process_map"][task_id] = process_id
            
            # Find performer role
            performer_role = (t.get("performer_role") or "").lower()
            if performer_role and performer_role in role_name_to_id:
                entities["task_role_map"][task_id] = role_name_to_id[performer_role]
            
            if chunk_id:
                entities["entity_chunk_map"][task_id] = chunk_id
            
            # Store next/previous task info for later sequence flow creation
            if t.get("next_task"):
                task._next_task_name = t.get("next_task")
            if t.get("previous_task"):
                task._previous_task_name = t.get("previous_task")
        
        # Process explicit task-role mappings
        for mapping in extracted.task_role_mappings:
            task_name = (mapping.get("task_name") or "").lower()
            role_name = (mapping.get("role_name") or "").lower()
            
            # Find matching task and role
            for task in entities["tasks"]:
                if task.name.lower() == task_name or task_name in task.name.lower():
                    if role_name in role_name_to_id:
                        entities["task_role_map"][task.task_id] = role_name_to_id[role_name]
                        break
        
        # Process explicit task-process mappings
        for mapping in extracted.task_process_mappings:
            task_name = (mapping.get("task_name") or "").lower()
            process_name = (mapping.get("process_name") or "").lower()
            
            for task in entities["tasks"]:
                if task.name.lower() == task_name or task_name in task.name.lower():
                    if process_name in process_name_to_id:
                        task.process_id = process_name_to_id[process_name]
                        entities["task_process_map"][task.task_id] = process_name_to_id[process_name]
                        break
        
        # Convert gateways
        for g in extracted.gateways:
            gw_type_str = (g.get("gateway_type") or "exclusive").lower()
            gw_type = GatewayType.EXCLUSIVE
            if gw_type_str == "parallel":
                gw_type = GatewayType.PARALLEL
            elif gw_type_str == "inclusive":
                gw_type = GatewayType.INCLUSIVE
            
            gateway_id = generate_id()
            
            # Find parent process
            parent_process_name = (g.get("parent_process") or "").lower()
            process_id = process_name_to_id.get(parent_process_name, "")
            if not process_id and entities["processes"]:
                process_id = entities["processes"][0].proc_id
            
            gateway = Gateway(
                gateway_id=gateway_id,
                process_id=process_id,
                gateway_type=gw_type,
                condition=g.get("condition", ""),
                description=g.get("description", "")
            )
            entities["gateways"].append(gateway)
            
            if chunk_id:
                entities["entity_chunk_map"][gateway_id] = chunk_id
        
        # Convert events
        for e in extracted.events:
            event_type_str = (e.get("event_type") or "start").lower()
            event_type = EventType.START
            if event_type_str == "end":
                event_type = EventType.END
            elif event_type_str == "intermediate":
                event_type = EventType.INTERMEDIATE
            
            event_id = generate_id()
            
            # Find parent process
            parent_process_name = (e.get("parent_process") or "").lower()
            process_id = process_name_to_id.get(parent_process_name, "")
            if not process_id and entities["processes"]:
                process_id = entities["processes"][0].proc_id
            
            event = Event(
                event_id=event_id,
                process_id=process_id,
                event_type=event_type,
                name=e.get("name", "Event"),
                trigger=e.get("trigger", "")
            )
            entities["events"].append(event)
            
            if chunk_id:
                entities["entity_chunk_map"][event_id] = chunk_id
        
        # Convert decisions and rules with role linkage
        for d in extracted.decisions:
            decision_id = generate_id()
            decision = DMNDecision(
                decision_id=decision_id,
                name=d.get("name", "Decision"),
                description=d.get("description", ""),
                input_data=d.get("input_data", []),
                output_data=d.get("output_data", [])
            )
            entities["decisions"].append(decision)
            
            # Link decision to role
            related_role = (d.get("related_role") or "").lower()
            if related_role and related_role in role_name_to_id:
                role_id = role_name_to_id[related_role]
                if role_id not in entities["role_decision_map"]:
                    entities["role_decision_map"][role_id] = []
                entities["role_decision_map"][role_id].append(decision_id)
            
            if chunk_id:
                entities["entity_chunk_map"][decision_id] = chunk_id
        
        for r in extracted.rules:
            rule_id = generate_id()
            rule = DMNRule(
                rule_id=rule_id,
                decision_id=r.get("decision_id", ""),
                when=r.get("when", r.get("condition", "")),
                then=r.get("then", r.get("result", "")),
                confidence=r.get("confidence", 1.0)
            )
            entities["rules"].append(rule)
            
            if chunk_id:
                entities["entity_chunk_map"][rule_id] = chunk_id
        
        # Process sequence flows from extracted data
        for flow in extracted.sequence_flows:
            from_task_name = (flow.get("from_task") or "").lower()
            to_task_name = (flow.get("to_task") or "").lower()
            condition = flow.get("condition", "")
            
            from_task_id = None
            to_task_id = None
            
            # Find task IDs by name
            for task in entities["tasks"]:
                task_lower = task.name.lower()
                if from_task_name and (task_lower == from_task_name or from_task_name in task_lower):
                    from_task_id = task.task_id
                if to_task_name and (task_lower == to_task_name or to_task_name in task_lower):
                    to_task_id = task.task_id
            
            if from_task_id and to_task_id:
                entities["sequence_flows"].append({
                    "from_task_id": from_task_id,
                    "to_task_id": to_task_id,
                    "condition": condition
                })
        
        # Also create sequence flows from next_task/previous_task attributes
        for task in entities["tasks"]:
            if hasattr(task, '_next_task_name') and task._next_task_name:
                next_name = task._next_task_name.lower()
                for other_task in entities["tasks"]:
                    if other_task.name.lower() == next_name or next_name in other_task.name.lower():
                        entities["sequence_flows"].append({
                            "from_task_id": task.task_id,
                            "to_task_id": other_task.task_id,
                            "condition": ""
                        })
                        break
        
        # Create default sequence flows based on order (within same process)
        # Group tasks by process
        tasks_by_process = {}
        for task in entities["tasks"]:
            proc_id = task.process_id or "default"
            if proc_id not in tasks_by_process:
                tasks_by_process[proc_id] = []
            tasks_by_process[proc_id].append(task)
        
        # Create sequence flows for tasks without explicit flows
        existing_flows = set()
        for flow in entities["sequence_flows"]:
            existing_flows.add((flow["from_task_id"], flow["to_task_id"]))
        
        for proc_id, proc_tasks in tasks_by_process.items():
            # Sort by order
            sorted_tasks = sorted(proc_tasks, key=lambda t: t.order)
            
            for i in range(len(sorted_tasks) - 1):
                from_task = sorted_tasks[i]
                to_task = sorted_tasks[i + 1]
                
                # Only add if not already exists
                if (from_task.task_id, to_task.task_id) not in existing_flows:
                    entities["sequence_flows"].append({
                        "from_task_id": from_task.task_id,
                        "to_task_id": to_task.task_id,
                        "condition": ""
                    })
        
        return entities
