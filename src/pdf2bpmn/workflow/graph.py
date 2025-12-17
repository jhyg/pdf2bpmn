"""LangGraph workflow definition for PDF to BPMN conversion."""
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import GraphState
from ..models.entities import (
    Process, Task, Role, Gateway, Event, 
    Skill, DMNDecision, DMNRule, Evidence, Ambiguity,
    AmbiguityStatus, TaskType, generate_id
)
from ..extractors.pdf_extractor import PDFExtractor
from ..extractors.entity_extractor import EntityExtractor
from ..graph.neo4j_client import Neo4jClient
from ..graph.vector_search import VectorSearch
from ..generators.bpmn_generator import BPMNGenerator
from ..generators.dmn_generator import DMNGenerator
from ..generators.skill_generator import SkillGenerator
from ..config import Config


class PDF2BPMNWorkflow:
    """Orchestrates the PDF to BPMN conversion workflow."""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.entity_extractor = EntityExtractor()
        self.neo4j = Neo4jClient()
        self.vector_search = VectorSearch(self.neo4j)
        self.bpmn_generator = BPMNGenerator()
        self.dmn_generator = DMNGenerator()
        self.skill_generator = SkillGenerator()
        
        # Accumulated relationship maps
        self.task_role_map = {}  # task_id -> role_id
        self.task_process_map = {}  # task_id -> process_id
        self.role_decision_map = {}  # role_id -> [decision_ids]
        self.entity_chunk_map = {}  # entity_id -> chunk_id
        self.role_skill_map = {}  # role_id -> [skill_ids]
        self.sequence_flows = []  # list of {from_id, to_id, from_type, to_type, condition}
        self.all_gateways = []  # list of Gateway objects
        
        # Name -> ID mappings for lookup
        self.process_name_to_id = {}
        self.role_name_to_id = {}
        self.task_name_to_id = {}
    
    def ingest_pdf(self, state: GraphState) -> GraphState:
        """Node: Ingest PDF and extract document structure."""
        print("📄 Ingesting PDF documents...")
        
        documents = []
        sections = []
        chunks = []
        
        for pdf_path in state.get("pdf_paths", []):
            doc, doc_sections, doc_chunks = self.pdf_extractor.extract_document(pdf_path)
            documents.append(doc)
            sections.extend(doc_sections)
            chunks.extend(doc_chunks)
            
            # Store in Neo4j
            self.neo4j.create_document(doc)
            for section in doc_sections:
                self.neo4j.create_section(section)
        
        return {
            "documents": documents,
            "sections": sections,
            "reference_chunks": chunks,
            "current_step": "segment_sections"
        }
    
    def segment_sections(self, state: GraphState) -> GraphState:
        """Node: Process and embed sections."""
        print("📑 Segmenting and embedding sections...")
        
        chunks = state.get("reference_chunks", [])
        documents = state.get("documents", [])
        doc_id = documents[0].doc_id if documents else ""
        
        # Batch embed chunks (in smaller batches to avoid rate limits)
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            self.vector_search.batch_embed_chunks(batch)
            
            # Store in Neo4j and link to document
            for chunk in batch:
                self.neo4j.create_chunk(chunk)
                if doc_id:
                    self.neo4j.link_chunk_to_document(chunk.chunk_id, doc_id)
        
        return {
            "reference_chunks": chunks,
            "current_step": "extract_candidates"
        }
    
    def extract_candidates(self, state: GraphState) -> GraphState:
        """Node: Extract process/task/role candidates from sections."""
        print("🔍 Extracting candidate entities...")
        
        all_processes = []
        all_tasks = []
        all_roles = []
        all_gateways = []
        all_events = []
        all_decisions = []
        all_rules = []
        
        sections = state.get("sections", [])
        documents = state.get("documents", [])
        chunks = state.get("reference_chunks", [])
        doc_id = documents[0].doc_id if documents else ""
        
        # Create chunk index for linking
        chunk_by_page = {}
        for chunk in chunks:
            if chunk.page not in chunk_by_page:
                chunk_by_page[chunk.page] = []
            chunk_by_page[chunk.page].append(chunk)
        
        for section in sections:
            if not section.content or len(section.content.strip()) < 50:
                continue
            
            # Find relevant chunk for this section (for evidence linking)
            section_chunk_id = ""
            if section.page_from in chunk_by_page and chunk_by_page[section.page_from]:
                section_chunk_id = chunk_by_page[section.page_from][0].chunk_id
            
            # Extract entities from section content with existing context
            # 기존 프로세스/역할/태스크 목록을 LLM에 전달하여 동일 엔티티 식별 개선
            existing_process_names = list(self.process_name_to_id.keys())
            existing_role_names = list(self.role_name_to_id.keys())
            
            # 기존 태스크 정보 수집 (이름, 역할, 프로세스)
            existing_tasks_info = []
            for task in all_tasks:
                task_info = {"name": task.name, "order": task.order}
                # 태스크의 역할 찾기
                if task.task_id in self.task_role_map:
                    role_id = self.task_role_map[task.task_id]
                    for role_name, rid in self.role_name_to_id.items():
                        if rid == role_id:
                            task_info["role"] = role_name
                            break
                # 태스크의 프로세스 찾기
                if task.process_id:
                    for proc_name, pid in self.process_name_to_id.items():
                        if pid == task.process_id:
                            task_info["process"] = proc_name
                            break
                existing_tasks_info.append(task_info)
            
            extracted = self.entity_extractor.extract_from_text(
                section.content,
                existing_processes=existing_process_names,
                existing_roles=existing_role_names,
                existing_tasks=existing_tasks_info
            )
            
            # Convert to entity objects with relationships
            entities = self.entity_extractor.convert_to_entities(
                extracted, 
                doc_id,
                chunk_id=section_chunk_id,
                existing_processes=self.process_name_to_id,
                existing_roles=self.role_name_to_id
            )
            
            # Collect entities
            all_processes.extend(entities["processes"])
            all_tasks.extend(entities["tasks"])
            all_roles.extend(entities["roles"])
            all_gateways.extend(entities["gateways"])
            all_events.extend(entities["events"])
            all_decisions.extend(entities["decisions"])
            all_rules.extend(entities["rules"])
            
            # Accumulate relationship mappings
            self.task_role_map.update(entities.get("task_role_map", {}))
            self.task_process_map.update(entities.get("task_process_map", {}))
            self.entity_chunk_map.update(entities.get("entity_chunk_map", {}))
            
            # Accumulate sequence flows
            self.sequence_flows.extend(entities.get("sequence_flows", []))
            
            for role_id, decision_ids in entities.get("role_decision_map", {}).items():
                if role_id not in self.role_decision_map:
                    self.role_decision_map[role_id] = []
                self.role_decision_map[role_id].extend(decision_ids)
            
            # Update name -> ID mappings
            for proc in entities["processes"]:
                self.process_name_to_id[proc.name.lower()] = proc.proc_id
            for role in entities["roles"]:
                self.role_name_to_id[role.name.lower()] = role.role_id
            for task in entities["tasks"]:
                self.task_name_to_id[task.name.lower()] = task.task_id
        
        return {
            "processes": all_processes,
            "tasks": all_tasks,
            "roles": all_roles,
            "gateways": all_gateways,
            "events": all_events,
            "dmn_decisions": all_decisions,
            "dmn_rules": all_rules,
            "current_step": "normalize_entities"
        }
    
    def extract_candidates_with_progress(self, state: GraphState, progress_callback=None) -> GraphState:
        """Extract candidates with progress callback for frontend updates."""
        print("🔍 Extracting candidate entities with progress...")
        
        all_processes = []
        all_tasks = []
        all_roles = []
        all_gateways = []
        all_events = []
        all_decisions = []
        all_rules = []
        
        sections = state.get("sections", [])
        documents = state.get("documents", [])
        chunks = state.get("reference_chunks", [])
        doc_id = documents[0].doc_id if documents else ""
        
        # Filter valid sections
        valid_sections = [s for s in sections if s.content and len(s.content.strip()) >= 50]
        total_sections = len(valid_sections)
        
        # Create chunk index for linking
        chunk_by_page = {}
        for chunk in chunks:
            if chunk.page not in chunk_by_page:
                chunk_by_page[chunk.page] = []
            chunk_by_page[chunk.page].append(chunk)
        
        for i, section in enumerate(valid_sections):
            # Report progress
            if progress_callback:
                section_preview = section.content[:50].replace('\n', ' ')
                progress_callback(
                    i + 1, 
                    total_sections, 
                    f"청크 {i+1}/{total_sections} LLM 분석 중: {section_preview}..."
                )
            
            # Find relevant chunk for this section
            section_chunk_id = ""
            if section.page_from in chunk_by_page and chunk_by_page[section.page_from]:
                section_chunk_id = chunk_by_page[section.page_from][0].chunk_id
            
            # Extract entities with existing context (프로세스, 역할, 태스크 모두 포함)
            existing_process_names = list(self.process_name_to_id.keys())
            existing_role_names = list(self.role_name_to_id.keys())
            
            # 기존 태스크 정보 수집 (이름, 역할, 프로세스)
            existing_tasks_info = []
            for task in all_tasks:
                task_info = {"name": task.name, "order": task.order}
                # 태스크의 역할 찾기
                if task.task_id in self.task_role_map:
                    role_id = self.task_role_map[task.task_id]
                    for role_name, rid in self.role_name_to_id.items():
                        if rid == role_id:
                            task_info["role"] = role_name
                            break
                # 태스크의 프로세스 찾기
                if task.process_id:
                    for proc_name, pid in self.process_name_to_id.items():
                        if pid == task.process_id:
                            task_info["process"] = proc_name
                            break
                existing_tasks_info.append(task_info)
            
            try:
                extracted = self.entity_extractor.extract_from_text(
                    section.content,
                    existing_processes=existing_process_names,
                    existing_roles=existing_role_names,
                    existing_tasks=existing_tasks_info
                )
                
                # Convert to entity objects
                entities = self.entity_extractor.convert_to_entities(
                    extracted, 
                    doc_id,
                    chunk_id=section_chunk_id,
                    existing_processes=self.process_name_to_id,
                    existing_roles=self.role_name_to_id
                )
                
                # Collect entities
                all_processes.extend(entities["processes"])
                all_tasks.extend(entities["tasks"])
                all_roles.extend(entities["roles"])
                all_gateways.extend(entities["gateways"])
                all_events.extend(entities["events"])
                all_decisions.extend(entities["decisions"])
                all_rules.extend(entities["rules"])
                
                # Accumulate mappings
                self.task_role_map.update(entities.get("task_role_map", {}))
                self.task_process_map.update(entities.get("task_process_map", {}))
                self.entity_chunk_map.update(entities.get("entity_chunk_map", {}))
                self.sequence_flows.extend(entities.get("sequence_flows", []))
                
                for role_id, decision_ids in entities.get("role_decision_map", {}).items():
                    if role_id not in self.role_decision_map:
                        self.role_decision_map[role_id] = []
                    self.role_decision_map[role_id].extend(decision_ids)
                
                # Update name mappings
                for proc in entities["processes"]:
                    self.process_name_to_id[proc.name.lower()] = proc.proc_id
                for role in entities["roles"]:
                    self.role_name_to_id[role.name.lower()] = role.role_id
                for task in entities["tasks"]:
                    self.task_name_to_id[task.name.lower()] = task.task_id
                    
            except Exception as e:
                print(f"   ⚠️ 청크 {i+1} 처리 중 오류: {e}")
                continue
        
        return {
            "processes": all_processes,
            "tasks": all_tasks,
            "roles": all_roles,
            "gateways": all_gateways,
            "events": all_events,
            "dmn_decisions": all_decisions,
            "dmn_rules": all_rules,
            "current_step": "normalize_entities"
        }
    
    def normalize_entities(self, state: GraphState) -> GraphState:
        """Node: Normalize and deduplicate entities using vector search."""
        print("🔄 Normalizing and deduplicating entities...")
        
        processes = state.get("processes", [])
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        gateways = state.get("gateways", [])
        events = state.get("events", [])
        decisions = state.get("dmn_decisions", [])
        
        # 1. 먼저 프로세스 병합 (같은 이름의 프로세스를 하나로)
        unique_processes, process_id_mapping = self._merge_duplicate_processes(processes)
        
        # 2. 태스크의 process_id를 병합된 프로세스로 업데이트
        tasks = self._update_task_process_ids(tasks, process_id_mapping)
        
        # 3. 게이트웨이, 이벤트의 process_id도 업데이트
        gateways = self._update_entity_process_ids(gateways, process_id_mapping, "gateway_id")
        events = self._update_entity_process_ids(events, process_id_mapping, "event_id")
        
        # 4. task_process_map도 업데이트
        self._update_task_process_map(process_id_mapping)
        
        # 5. 유사한 태스크 병합 (같은 역할의 연속 업무)
        tasks, task_id_mapping = self._merge_similar_tasks(tasks)
        print(f"   Tasks after merge: {len(tasks)}")
        
        # Deduplicate tasks
        unique_tasks = self._deduplicate_entities(tasks, "Task")
        
        # Deduplicate roles
        unique_roles = self._deduplicate_entities(roles, "Role")
        
        # Deduplicate decisions
        unique_decisions = self._deduplicate_entities(decisions, "Decision")
        
        print(f"   Processes: {len(processes)} → {len(unique_processes)} (merged {len(processes) - len(unique_processes)})")
        print(f"   Tasks: {len(tasks)} → {len(unique_tasks)}")
        print(f"   Roles: {len(roles)} → {len(unique_roles)}")
        print(f"   Decisions: {len(decisions)} → {len(unique_decisions)}")
        
        # Store in Neo4j
        for proc in unique_processes:
            self.neo4j.create_process(proc)
        
        for task in unique_tasks:
            self.neo4j.create_task(task)
        
        for role in unique_roles:
            self.neo4j.create_role(role)
        
        for gateway in gateways:
            self.neo4j.create_gateway(gateway)
        
        for event in events:
            self.neo4j.create_event(event)
        
        for decision in unique_decisions:
            self.neo4j.create_decision(decision)
        
        for rule in state.get("dmn_rules", []):
            self.neo4j.create_rule(rule)
        
        # Create relationships in batch
        print("🔗 Creating entity relationships...")
        # Only create evidence links if not disabled
        evidence_map = {} if Config.EVIDENCE_MODE == "off" else self.entity_chunk_map
        self.neo4j.create_all_relationships(
            task_role_map=self.task_role_map,
            task_process_map=self.task_process_map,
            role_decision_map=self.role_decision_map,
            entity_chunk_map=evidence_map
        )
        
        # Store gateways for sequence flow creation
        self.all_gateways = gateways
        
        # Create sequence flows (NEXT relationships between tasks/gateways)
        print("➡️ Creating sequence flows...")
        self._create_sequence_flows(unique_tasks, unique_processes)
        
        # Infer missing Task-Role relationships based on name matching
        self._infer_task_role_relationships(unique_tasks, unique_roles)
        
        # Infer missing Task-Process relationships 
        self._infer_task_process_relationships(unique_tasks, unique_processes)
        
        return {
            "processes": unique_processes,
            "tasks": unique_tasks,
            "roles": unique_roles,
            "dmn_decisions": unique_decisions,
            "current_step": "detect_ambiguities"
        }
    
    def _create_sequence_flows(self, tasks: list, processes: list):
        """Create NEXT relationships between tasks/gateways based on extracted and inferred sequence flows."""
        created_flows = set()
        
        # Build ID sets for validation
        task_ids = {t.task_id for t in tasks}
        gateway_ids = {g.gateway_id for g in self.all_gateways} if hasattr(self, 'all_gateways') else set()
        
        # First, create explicit sequence flows from extraction
        for flow in self.sequence_flows:
            # Support both old format (from_task_id) and new format (from_id)
            from_id = flow.get("from_id") or flow.get("from_task_id")
            to_id = flow.get("to_id") or flow.get("to_task_id")
            from_type = flow.get("from_type", "task")
            to_type = flow.get("to_type", "task")
            condition = flow.get("condition", "") or ""
            
            if from_id and to_id and (from_id, to_id) not in created_flows:
                # Create the appropriate relationship based on types
                if from_type == "gateway" and to_type == "task":
                    self.neo4j.link_gateway_to_task(from_id, to_id, condition)
                elif from_type == "task" and to_type == "gateway":
                    self.neo4j.link_task_to_gateway(from_id, to_id)
                else:
                    # Task to Task
                    self.neo4j.link_task_sequence(from_id, to_id, condition)
                
                created_flows.add((from_id, to_id))
                
                if condition:
                    print(f"   ✓ Flow with condition: {from_type}:{from_id[:8]} → {to_type}:{to_id[:8]} [{condition}]")
        
        # Group tasks by process
        tasks_by_process = {}
        for task in tasks:
            proc_id = task.process_id or "default"
            if proc_id not in tasks_by_process:
                tasks_by_process[proc_id] = []
            tasks_by_process[proc_id].append(task)
        
        # Create sequence flows for each process based on task order
        for proc_id, proc_tasks in tasks_by_process.items():
            sorted_tasks = sorted(proc_tasks, key=lambda t: t.order)
            
            for i in range(len(sorted_tasks) - 1):
                from_task = sorted_tasks[i]
                to_task = sorted_tasks[i + 1]
                
                if (from_task.task_id, to_task.task_id) not in created_flows:
                    self.neo4j.link_task_sequence(from_task.task_id, to_task.task_id)
                    created_flows.add((from_task.task_id, to_task.task_id))
        
        # Also use Neo4j to create sequences for each process
        for proc in processes:
            self.neo4j.create_task_sequence_for_process(proc.proc_id)
        
        print(f"   Created {len(created_flows)} sequence flows (NEXT relationships)")
    
    def _infer_task_role_relationships(self, tasks: list, roles: list):
        """Infer Task-Role relationships based on task descriptions."""
        role_keywords = {}
        for role in roles:
            keywords = [role.name.lower()]
            if role.org_unit:
                keywords.append(role.org_unit.lower())
            role_keywords[role.role_id] = keywords
        
        for task in tasks:
            if task.task_id in self.task_role_map:
                continue  # Already has a role
            
            task_text = (task.name + " " + task.description).lower()
            
            for role_id, keywords in role_keywords.items():
                for keyword in keywords:
                    if keyword in task_text and len(keyword) > 2:
                        self.neo4j.link_task_to_role(task.task_id, role_id)
                        self.task_role_map[task.task_id] = role_id
                        break
                if task.task_id in self.task_role_map:
                    break
    
    def _infer_task_process_relationships(self, tasks: list, processes: list):
        """Ensure all tasks are linked to a process."""
        if not processes:
            return
        
        default_process_id = processes[0].proc_id
        
        for task in tasks:
            if not task.process_id:
                task.process_id = default_process_id
                self.neo4j.link_task_to_role  # Link via HAS_TASK
                with self.neo4j.session() as session:
                    session.run("""
                        MATCH (p:Process {proc_id: $proc_id})
                        MATCH (t:Task {task_id: $task_id})
                        MERGE (p)-[:HAS_TASK]->(t)
                    """, {"proc_id": default_process_id, "task_id": task.task_id})
    
    def _merge_duplicate_processes(self, processes: list) -> tuple[list, dict]:
        """
        같은 이름의 프로세스를 병합하고, 병합된 프로세스 ID 매핑을 반환.
        
        Returns:
            tuple: (병합된 프로세스 목록, {기존 process_id -> 병합된 process_id} 매핑)
        """
        if not processes:
            return [], {}
        
        # 이름별로 프로세스 그룹화
        name_to_processes = {}
        for proc in processes:
            name_key = proc.name.lower().strip()
            if name_key not in name_to_processes:
                name_to_processes[name_key] = []
            name_to_processes[name_key].append(proc)
        
        unique_processes = []
        process_id_mapping = {}  # old_id -> new_id
        
        for name_key, proc_group in name_to_processes.items():
            # 첫 번째 프로세스를 primary로 선택
            primary = proc_group[0]
            unique_processes.append(primary)
            
            # 나머지 프로세스의 정보를 primary에 병합
            for other in proc_group[1:]:
                # ID 매핑 저장 (다른 프로세스 ID -> primary ID)
                process_id_mapping[other.proc_id] = primary.proc_id
                
                # 설명 병합 (빈 경우에만)
                if other.description and not primary.description:
                    primary.description = other.description
                if other.purpose and not primary.purpose:
                    primary.purpose = other.purpose
                
                print(f"   🔗 프로세스 병합: '{other.name}' ({other.proc_id[:8]}...) → ({primary.proc_id[:8]}...)")
            
            # primary도 자기 자신으로 매핑 (일관성)
            process_id_mapping[primary.proc_id] = primary.proc_id
        
        return unique_processes, process_id_mapping
    
    def _update_task_process_ids(self, tasks: list, process_id_mapping: dict) -> list:
        """태스크의 process_id를 병합된 프로세스 ID로 업데이트."""
        for task in tasks:
            if task.process_id and task.process_id in process_id_mapping:
                old_id = task.process_id
                new_id = process_id_mapping[old_id]
                if old_id != new_id:
                    task.process_id = new_id
        return tasks
    
    def _update_entity_process_ids(self, entities: list, process_id_mapping: dict, id_field: str) -> list:
        """게이트웨이, 이벤트 등의 process_id를 병합된 프로세스 ID로 업데이트."""
        for entity in entities:
            if hasattr(entity, 'process_id') and entity.process_id:
                if entity.process_id in process_id_mapping:
                    old_id = entity.process_id
                    new_id = process_id_mapping[old_id]
                    if old_id != new_id:
                        entity.process_id = new_id
        return entities
    
    def _update_task_process_map(self, process_id_mapping: dict):
        """task_process_map의 process_id를 병합된 ID로 업데이트."""
        for task_id, proc_id in list(self.task_process_map.items()):
            if proc_id in process_id_mapping:
                new_proc_id = process_id_mapping[proc_id]
                if proc_id != new_proc_id:
                    self.task_process_map[task_id] = new_proc_id
    
    def _merge_similar_tasks(self, tasks: list) -> tuple[list, dict]:
        """유사한 태스크를 병합합니다.
        
        병합 기준:
        1. 같은 프로세스 내에서
        2. 같은 역할이 수행하며
        3. 이름이 유사하거나 하나가 다른 하나를 포함하는 경우
        
        Returns:
            tuple: (병합된 태스크 리스트, {old_task_id: new_task_id} 매핑)
        """
        if not tasks:
            return tasks, {}
        
        task_id_mapping = {}  # old_id -> new_id
        
        # 프로세스별로 그룹화
        tasks_by_process = {}
        for task in tasks:
            proc_id = task.process_id or "no_process"
            if proc_id not in tasks_by_process:
                tasks_by_process[proc_id] = []
            tasks_by_process[proc_id].append(task)
        
        merged_tasks = []
        
        for proc_id, proc_tasks in tasks_by_process.items():
            # 역할별로 그룹화
            tasks_by_role = {}
            for task in proc_tasks:
                role_id = self.task_role_map.get(task.task_id, "no_role")
                if role_id not in tasks_by_role:
                    tasks_by_role[role_id] = []
                tasks_by_role[role_id].append(task)
            
            for role_id, role_tasks in tasks_by_role.items():
                # 같은 역할의 태스크들 중 유사한 것들 병합
                merged_role_tasks = self._merge_tasks_by_similarity(role_tasks, task_id_mapping)
                merged_tasks.extend(merged_role_tasks)
        
        # task_role_map 업데이트
        for old_id, new_id in task_id_mapping.items():
            if old_id in self.task_role_map and old_id != new_id:
                self.task_role_map[new_id] = self.task_role_map[old_id]
        
        return merged_tasks, task_id_mapping
    
    def _merge_tasks_by_similarity(self, tasks: list, task_id_mapping: dict) -> list:
        """이름 유사도를 기반으로 태스크를 병합합니다."""
        if len(tasks) <= 1:
            return tasks
        
        # order로 정렬
        sorted_tasks = sorted(tasks, key=lambda t: t.order)
        merged = []
        skip_indices = set()
        
        for i, task in enumerate(sorted_tasks):
            if i in skip_indices:
                continue
            
            task_name = task.name.lower().strip()
            merged_with = []
            
            # 다른 태스크와 비교
            for j, other_task in enumerate(sorted_tasks):
                if i == j or j in skip_indices:
                    continue
                
                other_name = other_task.name.lower().strip()
                
                # 병합 조건 체크
                should_merge = False
                
                # 1. 한쪽이 다른 쪽을 포함
                if task_name in other_name or other_name in task_name:
                    should_merge = True
                
                # 2. 핵심 단어가 같은 경우 (ex: "구매요청서 접수" vs "구매요청서 접수 및 검토")
                elif self._have_same_core_words(task_name, other_name):
                    should_merge = True
                
                # 3. 연속된 order이고 이름이 매우 유사
                elif abs(task.order - other_task.order) <= 1:
                    similarity = self._calc_name_similarity(task_name, other_name)
                    if similarity > 0.6:
                        should_merge = True
                
                if should_merge:
                    merged_with.append((j, other_task))
                    skip_indices.add(j)
            
            # 병합 수행
            if merged_with:
                # 가장 긴 이름을 가진 태스크를 대표로 선택
                all_related = [task] + [t for _, t in merged_with]
                representative = max(all_related, key=lambda t: len(t.name))
                
                # 설명 통합
                descriptions = [t.description for t in all_related if t.description]
                if descriptions:
                    representative.description = " | ".join(set(descriptions))
                
                # ID 매핑 기록
                for t in all_related:
                    if t.task_id != representative.task_id:
                        task_id_mapping[t.task_id] = representative.task_id
                
                merged.append(representative)
                print(f"   🔀 병합: {[t.name for t in all_related]} → {representative.name}")
            else:
                merged.append(task)
        
        return merged
    
    def _have_same_core_words(self, name1: str, name2: str) -> bool:
        """두 이름이 핵심 단어를 공유하는지 확인합니다."""
        # 한국어 조사/어미 제거를 위한 간단한 처리
        stop_words = {'및', '의', '을', '를', '이', '가', '에', '로', '으로', '와', '과', '에서', '부터', '까지'}
        
        words1 = set(name1.replace(' ', '').replace('및', ' ').split()) - stop_words
        words2 = set(name2.replace(' ', '').replace('및', ' ').split()) - stop_words
        
        # 공통 단어가 2개 이상이면 유사
        common = words1 & words2
        return len(common) >= 1 and len(common) >= min(len(words1), len(words2)) * 0.5
    
    def _calc_name_similarity(self, name1: str, name2: str) -> float:
        """두 이름의 유사도를 계산합니다 (0~1)."""
        # 간단한 Jaccard similarity
        set1 = set(name1)
        set2 = set(name2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_entities(self, entities: list, entity_type: str) -> list:
        """Deduplicate entities based on name similarity."""
        seen_names = {}
        unique = []
        
        for entity in entities:
            name = entity.name.lower().strip()
            
            if name in seen_names:
                continue
            
            # Check for similar existing entities
            try:
                match, score, action = self.vector_search.find_similar_entity(
                    entity_type, entity.name, getattr(entity, 'description', '')
                )
                
                if action == "merge" and match:
                    continue
            except:
                pass
            
            seen_names[name] = entity
            unique.append(entity)
        
        return unique
    
    def detect_ambiguities(self, state: GraphState) -> GraphState:
        """Node: Detect ambiguities that need human resolution."""
        print("❓ Detecting ambiguities...")
        
        questions = []
        
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        processes = state.get("processes", [])
        gateways = state.get("gateways", [])
        
        # Count tasks without roles
        tasks_without_roles = [t for t in tasks if t.task_id not in self.task_role_map]
        
        if tasks_without_roles and roles:
            # Create batch question for role assignment
            role_names = [r.name for r in roles]
            for task in tasks_without_roles[:10]:  # Limit to 10 questions
                questions.append(Ambiguity(
                    amb_id=generate_id(),
                    entity_type="Task",
                    entity_id=task.task_id,
                    question=f"'{task.name}' 태스크의 담당 역할은 누구인가요?",
                    options=role_names + ["새 역할 추가", "미정"],
                    status=AmbiguityStatus.OPEN
                ))
        
        # Store ambiguities in Neo4j
        for q in questions:
            self.neo4j.create_ambiguity(q)
        
        print(f"   {len(questions)} questions generated")
        print(f"   Tasks without roles: {len(tasks_without_roles)}")
        
        return {
            "open_questions": questions,
            "current_step": "ask_user" if questions else "generate_skills"
        }
    
    def ask_user(self, state: GraphState) -> GraphState:
        """Node: Wait for user input on ambiguities (HITL interrupt point)."""
        print("🙋 Waiting for user input...")
        
        questions = state.get("open_questions", [])
        open_questions = [q for q in questions if q.status == AmbiguityStatus.OPEN]
        
        if open_questions:
            current = open_questions[0]
            return {
                "current_question": current,
                "current_step": "waiting_for_user"
            }
        
        return {
            "current_question": None,
            "current_step": "generate_skills"
        }
    
    def apply_user_answer(self, state: GraphState) -> GraphState:
        """Node: Apply user's answer to resolve ambiguity."""
        print("✅ Applying user answer...")
        
        current_question = state.get("current_question")
        user_answer = state.get("user_answer")
        
        if current_question and user_answer:
            current_question.status = AmbiguityStatus.RESOLVED
            current_question.answer = user_answer
            
            self.neo4j.resolve_ambiguity(current_question.amb_id, user_answer)
            
            # Apply answer - link task to selected role
            if current_question.entity_type == "Task":
                role_name = user_answer.lower()
                if role_name in self.role_name_to_id:
                    role_id = self.role_name_to_id[role_name]
                    self.neo4j.link_task_to_role(current_question.entity_id, role_id)
                    self.task_role_map[current_question.entity_id] = role_id
            
            resolved = state.get("resolved_questions", [])
            resolved.append(current_question)
            
            open_questions = [
                q for q in state.get("open_questions", [])
                if q.amb_id != current_question.amb_id
            ]
            
            return {
                "resolved_questions": resolved,
                "open_questions": open_questions,
                "current_question": None,
                "user_answer": None,
                "current_step": "detect_ambiguities"
            }
        
        return {"current_step": "generate_skills"}
    
    def generate_skills(self, state: GraphState) -> GraphState:
        """Node: Generate skill documents for agent tasks."""
        print("📝 Generating skill documents...")
        
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        skills = []
        skill_docs = {}
        
        for task in tasks:
            if task.task_type == TaskType.AGENT:
                skill, markdown = self.skill_generator.generate_from_task(task)
                
                safe_name = "".join(
                    c if c.isalnum() or c in "._-" else "_" 
                    for c in task.name
                )
                filename = f"{safe_name}.skill.md"
                filepath = Config.OUTPUT_DIR / filename
                
                self.skill_generator.save(markdown, str(filepath))
                skill.md_path = str(filepath)
                skill_docs[task.task_id] = markdown
                
                self.neo4j.create_skill(skill)
                self.neo4j.link_task_to_skill(task.task_id, skill.skill_id)
                
                # Link skill to role if task has a role
                if task.task_id in self.task_role_map:
                    role_id = self.task_role_map[task.task_id]
                    self.neo4j.link_role_to_skill(role_id, skill.skill_id)
                    
                    if role_id not in self.role_skill_map:
                        self.role_skill_map[role_id] = []
                    self.role_skill_map[role_id].append(skill.skill_id)
                
                skills.append(skill)
        
        print(f"   Generated {len(skills)} skill documents")
        
        return {
            "skills": skills,
            "skill_docs": skill_docs,
            "current_step": "generate_dmn"
        }
    
    def generate_dmn(self, state: GraphState) -> GraphState:
        """Node: Generate DMN decision tables."""
        print("📊 Generating DMN decision tables...")
        
        decisions = state.get("dmn_decisions", [])
        rules = state.get("dmn_rules", [])
        
        if decisions:
            dmn_xml = self.dmn_generator.generate(decisions, rules)
            
            dmn_path = Config.OUTPUT_DIR / "decisions.dmn"
            self.dmn_generator.save(dmn_xml, str(dmn_path))
            
            dmn_json = self.dmn_generator.generate_json(decisions, rules)
            json_path = Config.OUTPUT_DIR / "decisions.json"
            self.dmn_generator.save(dmn_json, str(json_path))
            
            # Link decisions to roles that make them
            for role_id, decision_ids in self.role_decision_map.items():
                for decision_id in decision_ids:
                    self.neo4j.link_role_to_decision(role_id, decision_id)
            
            print(f"   Generated {len(decisions)} decision tables")
            
            return {
                "dmn_xml": dmn_xml,
                "current_step": "assemble_bpmn"
            }
        
        return {
            "dmn_xml": None,
            "current_step": "assemble_bpmn"
        }
    
    def assemble_bpmn(self, state: GraphState) -> GraphState:
        """Node: Assemble final BPMN XML - one per process."""
        print("🔧 Assembling BPMN XML...")
        
        processes = state.get("processes", [])
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        gateways = state.get("gateways", [])
        events = state.get("events", [])
        
        if not processes:
            # Create a default process if none exists
            default_process = Process(
                name="Extracted Process",
                purpose="Automatically extracted from document",
                description="Process extracted from PDF document"
            )
            processes = [default_process]
            # Assign all unassigned tasks to the default process
            for task in tasks:
                if not task.process_id:
                    task.process_id = default_process.proc_id
        
        # Generate BPMN for each process
        bpmn_xmls = {}  # process_id -> bpmn_xml
        bpmn_files = {}  # process_id -> file_path
        
        print(f"   Processing {len(processes)} process(es)...")
        
        for process in processes:
            print(f"\n   📋 Processing: {process.name} (ID: {process.proc_id})")
            
            # Filter entities by process_id
            process_tasks = [t for t in tasks if t.process_id == process.proc_id or (not t.process_id and process == processes[0])]
            process_gateways = [g for g in gateways if g.process_id == process.proc_id or (not g.process_id and process == processes[0])]
            process_events = [e for e in events if e.process_id == process.proc_id or (not e.process_id and process == processes[0])]
            
            # Assign unassigned entities to this process (only for first process)
            if process == processes[0]:
                for task in process_tasks:
                    if not task.process_id:
                        task.process_id = process.proc_id
                for gateway in process_gateways:
                    if not gateway.process_id:
                        gateway.process_id = process.proc_id
                for event in process_events:
                    if not event.process_id:
                        event.process_id = process.proc_id
            
            print(f"      Tasks: {len(process_tasks)}, Gateways: {len(process_gateways)}, Events: {len(process_events)}")
            
            # Get roles used by tasks in this process
            process_task_ids = {t.task_id for t in process_tasks}
            process_role_ids = set()
            for task_id, role_id in self.task_role_map.items():
                if task_id in process_task_ids:
                    process_role_ids.add(role_id)
            
            process_roles = [r for r in roles if r.role_id in process_role_ids]
            print(f"      Roles: {len(process_roles)}")
            
            # Get sequence flows from Neo4j for this process
            neo4j_sequence_flows = self.neo4j.get_sequence_flows(process.proc_id)
            print(f"      Sequence flows: {len(neo4j_sequence_flows)}")
            
            # Log flows with conditions
            flows_with_conditions = [f for f in neo4j_sequence_flows if f.get("condition")]
            if flows_with_conditions:
                print(f"      Conditional flows: {len(flows_with_conditions)}")
                for flow in flows_with_conditions[:3]:  # Show first 3
                    print(f"         {flow.get('from_name')} → {flow.get('to_name')}: {flow.get('condition')}")
            
            # Generate BPMN XML for this process
            bpmn_xml = self.bpmn_generator.generate(
                process=process,
                tasks=process_tasks,
                roles=process_roles,
                gateways=process_gateways,
                events=process_events,
                task_role_map=self.task_role_map,
                neo4j_sequence_flows=neo4j_sequence_flows
            )
            
            # Save to file with process-specific name
            # Use sanitized process name for filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in process.name)
            safe_name = safe_name.replace(' ', '_').replace('-', '_')[:40]  # Limit length
            # Use full proc_id (already UUID) for uniqueness
            bpmn_filename = f"process_{safe_name}_{process.proc_id}.bpmn"
            bpmn_path = Config.OUTPUT_DIR / bpmn_filename
            
            self.bpmn_generator.save(bpmn_xml, str(bpmn_path))
            print(f"      ✅ Saved: {bpmn_filename}")
            
            bpmn_xmls[process.proc_id] = bpmn_xml
            bpmn_files[process.proc_id] = str(bpmn_path)
        
        # For backward compatibility, keep the first BPMN as the main one
        main_bpmn_xml = bpmn_xmls.get(processes[0].proc_id) if processes else None
        
        # Also save the first one as process.bpmn for backward compatibility
        if main_bpmn_xml:
            default_bpmn_path = Config.OUTPUT_DIR / "process.bpmn"
            self.bpmn_generator.save(main_bpmn_xml, str(default_bpmn_path))
        
        return {
            "bpmn_xml": main_bpmn_xml,  # Backward compatibility
            "bpmn_xmls": bpmn_xmls,  # All BPMN XMLs keyed by process_id
            "bpmn_files": bpmn_files,  # File paths keyed by process_id
            "current_step": "validate_consistency"
        }
    
    def validate_consistency(self, state: GraphState) -> GraphState:
        """Node: Validate consistency of generated artifacts."""
        print("✔️ Validating consistency...")
        
        errors = []
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        
        task_role_coverage = len(self.task_role_map)
        if tasks and task_role_coverage < len(tasks) * 0.3:
            errors.append(f"경고: {len(tasks) - task_role_coverage}개 태스크에 역할이 지정되지 않았습니다.")
        
        if len(roles) == 0 and len(tasks) > 0:
            errors.append("역할(Role)이 추출되지 않았습니다.")
        
        if errors:
            for err in errors:
                print(f"   ⚠️ {err}")
            return {
                "error": "; ".join(errors),
                "current_step": "export_artifacts"
            }
        
        print("   ✅ All validations passed")
        return {
            "error": None,
            "current_step": "export_artifacts"
        }
    
    def export_artifacts(self, state: GraphState) -> GraphState:
        """Node: Export final artifacts."""
        print("📦 Exporting artifacts...")
        
        # Print relationship statistics
        print(f"\n📊 Relationship Statistics:")
        print(f"   - Task → Task (NEXT/Sequence): {len(self.sequence_flows)}")
        print(f"   - Task → Role (PERFORMED_BY): {len(self.task_role_map)}")
        print(f"   - Task → Process (HAS_TASK): {len(self.task_process_map)}")
        print(f"   - Role → Decision (MAKES_DECISION): {sum(len(v) for v in self.role_decision_map.values())}")
        print(f"   - Entity → Document (SUPPORTED_BY): {len(self.entity_chunk_map)}")
        print(f"   - Role → Skill (HAS_SKILL): {sum(len(v) for v in self.role_skill_map.values())}")
        
        output_summary = {
            "bpmn_path": str(Config.OUTPUT_DIR / "process.bpmn"),
            "dmn_path": str(Config.OUTPUT_DIR / "decisions.dmn") if state.get("dmn_xml") else None,
            "skill_count": len(state.get("skills", [])),
            "process_count": len(state.get("processes", [])),
            "task_count": len(state.get("tasks", [])),
            "role_count": len(state.get("roles", []))
        }
        
        print(f"\n✅ Export complete!")
        print(f"   - BPMN: {output_summary['bpmn_path']}")
        if output_summary['dmn_path']:
            print(f"   - DMN: {output_summary['dmn_path']}")
        print(f"   - Skills: {output_summary['skill_count']} documents")
        print(f"   - Processes: {output_summary['process_count']}")
        print(f"   - Tasks: {output_summary['task_count']}")
        print(f"   - Roles: {output_summary['role_count']}")
        
        return {
            "current_step": "completed"
        }


def should_ask_user(state: GraphState) -> Literal["ask_user", "generate_skills"]:
    """Routing function: determine if we need user input."""
    open_questions = state.get("open_questions", [])
    unresolved = [q for q in open_questions if q.status == AmbiguityStatus.OPEN]
    
    if unresolved:
        return "ask_user"
    return "generate_skills"


def has_more_questions(state: GraphState) -> Literal["ask_user", "generate_skills"]:
    """Check if there are more questions to ask."""
    open_questions = state.get("open_questions", [])
    unresolved = [q for q in open_questions if q.status == AmbiguityStatus.OPEN]
    
    if unresolved:
        return "ask_user"
    return "generate_skills"


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""
    
    workflow_handler = PDF2BPMNWorkflow()
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("ingest_pdf", workflow_handler.ingest_pdf)
    workflow.add_node("segment_sections", workflow_handler.segment_sections)
    workflow.add_node("extract_candidates", workflow_handler.extract_candidates)
    workflow.add_node("normalize_entities", workflow_handler.normalize_entities)
    workflow.add_node("detect_ambiguities", workflow_handler.detect_ambiguities)
    workflow.add_node("ask_user", workflow_handler.ask_user)
    workflow.add_node("apply_user_answer", workflow_handler.apply_user_answer)
    workflow.add_node("generate_skills", workflow_handler.generate_skills)
    workflow.add_node("generate_dmn", workflow_handler.generate_dmn)
    workflow.add_node("assemble_bpmn", workflow_handler.assemble_bpmn)
    workflow.add_node("validate_consistency", workflow_handler.validate_consistency)
    workflow.add_node("export_artifacts", workflow_handler.export_artifacts)
    
    workflow.set_entry_point("ingest_pdf")
    
    workflow.add_edge("ingest_pdf", "segment_sections")
    workflow.add_edge("segment_sections", "extract_candidates")
    workflow.add_edge("extract_candidates", "normalize_entities")
    workflow.add_edge("normalize_entities", "detect_ambiguities")
    
    workflow.add_conditional_edges(
        "detect_ambiguities",
        should_ask_user,
        {
            "ask_user": "ask_user",
            "generate_skills": "generate_skills"
        }
    )
    
    workflow.add_edge("ask_user", "apply_user_answer")
    
    workflow.add_conditional_edges(
        "apply_user_answer",
        has_more_questions,
        {
            "ask_user": "ask_user",
            "generate_skills": "generate_skills"
        }
    )
    
    workflow.add_edge("generate_skills", "generate_dmn")
    workflow.add_edge("generate_dmn", "assemble_bpmn")
    workflow.add_edge("assemble_bpmn", "validate_consistency")
    workflow.add_edge("validate_consistency", "export_artifacts")
    workflow.add_edge("export_artifacts", END)
    
    return workflow


def compile_workflow_with_checkpointer():
    """Compile workflow with memory checkpointer for HITL support."""
    workflow = create_workflow()
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
