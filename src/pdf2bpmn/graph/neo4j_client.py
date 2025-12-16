"""Neo4j database client and schema management."""
from typing import Any, Optional
from contextlib import contextmanager

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from ..config import Config
from ..models.entities import (
    Document, Section, Process, Task, Role, Gateway, Event,
    Skill, DMNDecision, DMNRule, Evidence, Ambiguity,
    ReferenceChunk, ProcessDefFragment, Alias, Conflict
)


class Neo4jClient:
    """Client for Neo4j database operations."""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self._driver = None
    
    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self._driver
    
    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self):
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def verify_connection(self) -> bool:
        """Verify Neo4j connection is working."""
        try:
            with self.session() as session:
                session.run("RETURN 1")
            return True
        except ServiceUnavailable:
            return False
    
    def init_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        constraints = [
            # Document structure
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:ReferenceChunk) REQUIRE c.chunk_id IS UNIQUE",
            
            # Process hierarchy
            "CREATE CONSTRAINT proc_id IF NOT EXISTS FOR (p:Process) REQUIRE p.proc_id IS UNIQUE",
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE",
            "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (r:Role) REQUIRE r.role_id IS UNIQUE",
            "CREATE CONSTRAINT gateway_id IF NOT EXISTS FOR (g:Gateway) REQUIRE g.gateway_id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            
            # Skills and decisions
            "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.skill_id IS UNIQUE",
            "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:DMNDecision) REQUIRE d.decision_id IS UNIQUE",
            "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:DMNRule) REQUIRE r.rule_id IS UNIQUE",
            
            # Evidence and tracking
            "CREATE CONSTRAINT evi_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evi_id IS UNIQUE",
            "CREATE CONSTRAINT amb_id IF NOT EXISTS FOR (a:Ambiguity) REQUIRE a.amb_id IS UNIQUE",
            "CREATE CONSTRAINT alias_id IF NOT EXISTS FOR (a:Alias) REQUIRE a.alias_id IS UNIQUE",
            "CREATE CONSTRAINT conflict_id IF NOT EXISTS FOR (c:Conflict) REQUIRE c.conflict_id IS UNIQUE",
            "CREATE CONSTRAINT fragment_id IF NOT EXISTS FOR (f:ProcessDefFragment) REQUIRE f.fragment_id IS UNIQUE",
        ]
        
        indexes = [
            # Full-text search indexes
            "CREATE FULLTEXT INDEX process_name_idx IF NOT EXISTS FOR (p:Process) ON EACH [p.name, p.description]",
            "CREATE FULLTEXT INDEX task_name_idx IF NOT EXISTS FOR (t:Task) ON EACH [t.name, t.description]",
            "CREATE FULLTEXT INDEX role_name_idx IF NOT EXISTS FOR (r:Role) ON EACH [r.name, r.org_unit]",
            "CREATE FULLTEXT INDEX skill_name_idx IF NOT EXISTS FOR (s:Skill) ON EACH [s.name, s.summary]",
            
            # Vector index for embeddings (Neo4j 5.11+)
            """
            CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS
            FOR (c:ReferenceChunk)
            ON c.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """,
        ]
        
        with self.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Constraint warning: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Index warning: {e}")
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)."""
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    # ==================== Create Operations ====================
    
    def create_document(self, doc: Document) -> str:
        """Create a Document node."""
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.title = $title,
            d.source = $source,
            d.page_count = $page_count,
            d.uploaded_at = datetime($uploaded_at),
            d.version = $version,
            d.created_by = $created_by
        RETURN d.doc_id
        """
        with self.session() as session:
            result = session.run(query, {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "page_count": doc.page_count,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "version": doc.version,
                "created_by": doc.created_by
            })
            return result.single()[0]
    
    def create_section(self, section: Section) -> str:
        """Create a Section node and link to Document."""
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (s:Section {section_id: $section_id})
        SET s.heading = $heading,
            s.level = $level,
            s.page_from = $page_from,
            s.page_to = $page_to,
            s.content = $content
        MERGE (d)-[:HAS_SECTION]->(s)
        RETURN s.section_id
        """
        with self.session() as session:
            result = session.run(query, {
                "doc_id": section.doc_id,
                "section_id": section.section_id,
                "heading": section.heading,
                "level": section.level,
                "page_from": section.page_from,
                "page_to": section.page_to,
                "content": section.content[:5000] if section.content else ""
            })
            return result.single()[0]
    
    def create_chunk(self, chunk: ReferenceChunk) -> str:
        """Create a ReferenceChunk node."""
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c:ReferenceChunk {chunk_id: $chunk_id})
        SET c.page = $page,
            c.span = $span,
            c.text = $text,
            c.hash = $hash,
            c.embedding = $embedding
        MERGE (c)-[:FROM_DOCUMENT]->(d)
        RETURN c.chunk_id
        """
        with self.session() as session:
            result = session.run(query, {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "span": chunk.span,
                "text": chunk.text,
                "hash": chunk.hash,
                "embedding": chunk.embedding
            })
            return result.single()[0]
    
    def create_process(self, process: Process) -> str:
        """Create a Process node."""
        query = """
        MERGE (p:Process {proc_id: $proc_id})
        SET p.name = $name,
            p.purpose = $purpose,
            p.description = $description,
            p.triggers = $triggers,
            p.outcomes = $outcomes,
            p.version = $version,
            p.created_by = $created_by
        RETURN p.proc_id
        """
        with self.session() as session:
            result = session.run(query, {
                "proc_id": process.proc_id,
                "name": process.name,
                "purpose": process.purpose,
                "description": process.description,
                "triggers": process.triggers,
                "outcomes": process.outcomes,
                "version": process.version,
                "created_by": process.created_by
            })
            return result.single()[0]
    
    def create_task(self, task: Task) -> str:
        """Create a Task node and link to Process."""
        query = """
        MERGE (t:Task {task_id: $task_id})
        SET t.name = $name,
            t.task_type = $task_type,
            t.description = $description,
            t.order = $order,
            t.version = $version,
            t.created_by = $created_by
        WITH t
        OPTIONAL MATCH (p:Process {proc_id: $process_id})
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p)-[:HAS_TASK]->(t)
        )
        RETURN t.task_id
        """
        with self.session() as session:
            result = session.run(query, {
                "task_id": task.task_id,
                "process_id": task.process_id,
                "name": task.name,
                "task_type": task.task_type.value,
                "description": task.description,
                "order": task.order,
                "version": task.version,
                "created_by": task.created_by
            })
            return result.single()[0]
    
    def create_role(self, role: Role) -> str:
        """Create a Role node."""
        query = """
        MERGE (r:Role {role_id: $role_id})
        SET r.name = $name,
            r.org_unit = $org_unit,
            r.persona_hint = $persona_hint,
            r.version = $version
        RETURN r.role_id
        """
        with self.session() as session:
            result = session.run(query, {
                "role_id": role.role_id,
                "name": role.name,
                "org_unit": role.org_unit,
                "persona_hint": role.persona_hint,
                "version": role.version
            })
            return result.single()[0]
    
    def create_gateway(self, gateway: Gateway) -> str:
        """Create a Gateway node."""
        query = """
        MERGE (g:Gateway {gateway_id: $gateway_id})
        SET g.gateway_type = $gateway_type,
            g.condition = $condition,
            g.description = $description
        WITH g
        OPTIONAL MATCH (p:Process {proc_id: $process_id})
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p)-[:HAS_GATEWAY]->(g)
        )
        RETURN g.gateway_id
        """
        with self.session() as session:
            result = session.run(query, {
                "gateway_id": gateway.gateway_id,
                "process_id": gateway.process_id,
                "gateway_type": gateway.gateway_type.value,
                "condition": gateway.condition,
                "description": gateway.description
            })
            return result.single()[0]
    
    def create_event(self, event: Event) -> str:
        """Create an Event node."""
        query = """
        MERGE (e:Event {event_id: $event_id})
        SET e.event_type = $event_type,
            e.name = $name,
            e.trigger = $trigger
        WITH e
        OPTIONAL MATCH (p:Process {proc_id: $process_id})
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p)-[:HAS_EVENT]->(e)
        )
        RETURN e.event_id
        """
        with self.session() as session:
            result = session.run(query, {
                "event_id": event.event_id,
                "process_id": event.process_id,
                "event_type": event.event_type.value,
                "name": event.name,
                "trigger": event.trigger
            })
            return result.single()[0]
    
    def create_skill(self, skill: Skill) -> str:
        """Create a Skill node."""
        query = """
        MERGE (s:Skill {skill_id: $skill_id})
        SET s.name = $name,
            s.summary = $summary,
            s.purpose = $purpose,
            s.inputs = $inputs,
            s.outputs = $outputs,
            s.preconditions = $preconditions,
            s.procedure = $procedure,
            s.exceptions = $exceptions,
            s.tools = $tools,
            s.md_path = $md_path,
            s.version = $version
        RETURN s.skill_id
        """
        with self.session() as session:
            result = session.run(query, {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "summary": skill.summary,
                "purpose": skill.purpose,
                "inputs": str(skill.inputs),
                "outputs": str(skill.outputs),
                "preconditions": skill.preconditions,
                "procedure": skill.procedure,
                "exceptions": skill.exceptions,
                "tools": skill.tools,
                "md_path": skill.md_path,
                "version": skill.version
            })
            return result.single()[0]
    
    def create_decision(self, decision: DMNDecision) -> str:
        """Create a DMNDecision node."""
        query = """
        MERGE (d:DMNDecision {decision_id: $decision_id})
        SET d.name = $name,
            d.description = $description,
            d.input_data = $input_data,
            d.output_data = $output_data
        RETURN d.decision_id
        """
        with self.session() as session:
            result = session.run(query, {
                "decision_id": decision.decision_id,
                "name": decision.name,
                "description": decision.description,
                "input_data": decision.input_data,
                "output_data": decision.output_data
            })
            return result.single()[0]
    
    def create_rule(self, rule: DMNRule) -> str:
        """Create a DMNRule node and link to Decision."""
        query = """
        MERGE (r:DMNRule {rule_id: $rule_id})
        SET r.when_condition = $when_condition,
            r.then_result = $then_result,
            r.confidence = $confidence
        WITH r
        OPTIONAL MATCH (d:DMNDecision {decision_id: $decision_id})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:HAS_RULE]->(r)
        )
        RETURN r.rule_id
        """
        with self.session() as session:
            result = session.run(query, {
                "rule_id": rule.rule_id,
                "decision_id": rule.decision_id,
                "when_condition": rule.when,
                "then_result": rule.then,
                "confidence": rule.confidence
            })
            return result.single()[0]
    
    def create_ambiguity(self, ambiguity: Ambiguity) -> str:
        """Create an Ambiguity node for HITL questions."""
        query = """
        MERGE (a:Ambiguity {amb_id: $amb_id})
        SET a.entity_type = $entity_type,
            a.entity_id = $entity_id,
            a.question = $question,
            a.options = $options,
            a.status = $status,
            a.answer = $answer
        RETURN a.amb_id
        """
        with self.session() as session:
            result = session.run(query, {
                "amb_id": ambiguity.amb_id,
                "entity_type": ambiguity.entity_type,
                "entity_id": ambiguity.entity_id,
                "question": ambiguity.question,
                "options": ambiguity.options,
                "status": ambiguity.status.value,
                "answer": ambiguity.answer
            })
            return result.single()[0]
    
    def create_evidence_link(
        self, 
        entity_type: str, 
        entity_id: str, 
        chunk_id: str
    ):
        """Create SUPPORTED_BY relationship between entity and chunk."""
        query = f"""
        MATCH (e:{entity_type} {{{entity_type.lower()}_id: $entity_id}})
        MATCH (c:ReferenceChunk {{chunk_id: $chunk_id}})
        MERGE (e)-[:SUPPORTED_BY]->(c)
        """
        # Handle different ID field names
        id_field_map = {
            "Process": "proc_id",
            "Task": "task_id",
            "Role": "role_id",
            "Gateway": "gateway_id",
            "Event": "event_id",
            "Skill": "skill_id",
            "DMNDecision": "decision_id",
            "DMNRule": "rule_id"
        }
        id_field = id_field_map.get(entity_type, f"{entity_type.lower()}_id")
        
        query = f"""
        MATCH (e:{entity_type} {{{id_field}: $entity_id}})
        MATCH (c:ReferenceChunk {{chunk_id: $chunk_id}})
        MERGE (e)-[:SUPPORTED_BY]->(c)
        """
        with self.session() as session:
            session.run(query, {
                "entity_id": entity_id,
                "chunk_id": chunk_id
            })
    
    def link_task_to_role(self, task_id: str, role_id: str):
        """Create PERFORMED_BY relationship between Task and Role."""
        query = """
        MATCH (t:Task {task_id: $task_id})
        MATCH (r:Role {role_id: $role_id})
        MERGE (t)-[:PERFORMED_BY]->(r)
        """
        with self.session() as session:
            session.run(query, {"task_id": task_id, "role_id": role_id})
    
    def link_task_to_skill(self, task_id: str, skill_id: str):
        """Create USES_SKILL relationship between Task and Skill."""
        query = """
        MATCH (t:Task {task_id: $task_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (t)-[:USES_SKILL]->(s)
        """
        with self.session() as session:
            session.run(query, {"task_id": task_id, "skill_id": skill_id})
    
    def link_process_to_decision(self, proc_id: str, decision_id: str):
        """Create USES_DECISION relationship."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MERGE (p)-[:USES_DECISION]->(d)
        """
        with self.session() as session:
            session.run(query, {"proc_id": proc_id, "decision_id": decision_id})
    
    def link_role_to_skill(self, role_id: str, skill_id: str):
        """Create HAS_SKILL relationship between Role and Skill."""
        query = """
        MATCH (r:Role {role_id: $role_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (r)-[:HAS_SKILL]->(s)
        """
        with self.session() as session:
            session.run(query, {"role_id": role_id, "skill_id": skill_id})
    
    def link_role_to_decision(self, role_id: str, decision_id: str):
        """Create MAKES_DECISION relationship between Role and DMNDecision."""
        query = """
        MATCH (r:Role {role_id: $role_id})
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MERGE (r)-[:MAKES_DECISION]->(d)
        """
        with self.session() as session:
            session.run(query, {"role_id": role_id, "decision_id": decision_id})
    
    def link_skill_to_decision(self, skill_id: str, decision_id: str):
        """Create USES_DECISION relationship between Skill and DMNDecision."""
        query = """
        MATCH (s:Skill {skill_id: $skill_id})
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MERGE (s)-[:USES_DECISION]->(d)
        """
        with self.session() as session:
            session.run(query, {"skill_id": skill_id, "decision_id": decision_id})
    
    def link_chunk_to_document(self, chunk_id: str, doc_id: str):
        """Create FROM_DOCUMENT relationship between ReferenceChunk and Document."""
        query = """
        MATCH (c:ReferenceChunk {chunk_id: $chunk_id})
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c)-[:FROM_DOCUMENT]->(d)
        """
        with self.session() as session:
            session.run(query, {"chunk_id": chunk_id, "doc_id": doc_id})
    
    def link_task_sequence(self, from_task_id: str, to_task_id: str, condition: str = None):
        """Create NEXT (sequence flow) relationship between Tasks."""
        query = """
        MATCH (t1:Task {task_id: $from_task_id})
        MATCH (t2:Task {task_id: $to_task_id})
        MERGE (t1)-[r:NEXT]->(t2)
        SET r.condition = $condition
        """
        with self.session() as session:
            session.run(query, {
                "from_task_id": from_task_id,
                "to_task_id": to_task_id,
                "condition": condition
            })
    
    def link_gateway_to_task(self, gateway_id: str, task_id: str, condition: str = None, is_incoming: bool = False):
        """Create flow relationship between Gateway and Task."""
        if is_incoming:
            query = """
            MATCH (t:Task {task_id: $task_id})
            MATCH (g:Gateway {gateway_id: $gateway_id})
            MERGE (t)-[r:NEXT]->(g)
            """
        else:
            query = """
            MATCH (g:Gateway {gateway_id: $gateway_id})
            MATCH (t:Task {task_id: $task_id})
            MERGE (g)-[r:NEXT {condition: $condition}]->(t)
            """
        with self.session() as session:
            session.run(query, {
                "gateway_id": gateway_id,
                "task_id": task_id,
                "condition": condition
            })
    
    def link_event_to_task(self, event_id: str, task_id: str, is_start: bool = True):
        """Create flow relationship between Event and Task."""
        if is_start:
            query = """
            MATCH (e:Event {event_id: $event_id})
            MATCH (t:Task {task_id: $task_id})
            MERGE (e)-[:NEXT]->(t)
            """
        else:
            query = """
            MATCH (t:Task {task_id: $task_id})
            MATCH (e:Event {event_id: $event_id})
            MERGE (t)-[:NEXT]->(e)
            """
        with self.session() as session:
            session.run(query, {"event_id": event_id, "task_id": task_id})
    
    def create_task_sequence_for_process(self, proc_id: str):
        """Create NEXT relationships between tasks in a process based on order."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(t:Task)
        WITH t ORDER BY t.order
        WITH collect(t) as tasks
        UNWIND range(0, size(tasks)-2) as i
        WITH tasks[i] as t1, tasks[i+1] as t2
        MERGE (t1)-[:NEXT]->(t2)
        """
        with self.session() as session:
            session.run(query, {"proc_id": proc_id})
    
    def create_all_relationships(
        self,
        task_role_map: dict,
        task_process_map: dict,
        role_decision_map: dict,
        entity_chunk_map: dict,
        role_skill_map: dict = None
    ):
        """Create all relationships in batch."""
        with self.session() as session:
            # Task -> Role (PERFORMED_BY)
            for task_id, role_id in task_role_map.items():
                session.run("""
                    MATCH (t:Task {task_id: $task_id})
                    MATCH (r:Role {role_id: $role_id})
                    MERGE (t)-[:PERFORMED_BY]->(r)
                """, {"task_id": task_id, "role_id": role_id})
            
            # Task -> Process (belongs to, via HAS_TASK from Process)
            for task_id, proc_id in task_process_map.items():
                session.run("""
                    MATCH (p:Process {proc_id: $proc_id})
                    MATCH (t:Task {task_id: $task_id})
                    MERGE (p)-[:HAS_TASK]->(t)
                """, {"task_id": task_id, "proc_id": proc_id})
            
            # Role -> DMNDecision (MAKES_DECISION)
            for role_id, decision_ids in role_decision_map.items():
                for decision_id in decision_ids:
                    session.run("""
                        MATCH (r:Role {role_id: $role_id})
                        MATCH (d:DMNDecision {decision_id: $decision_id})
                        MERGE (r)-[:MAKES_DECISION]->(d)
                    """, {"role_id": role_id, "decision_id": decision_id})
            
            # Entity -> ReferenceChunk (SUPPORTED_BY) for evidence
            id_field_map = {
                "Process": "proc_id",
                "Task": "task_id",
                "Role": "role_id",
                "Gateway": "gateway_id",
                "Event": "event_id",
                "Skill": "skill_id",
                "DMNDecision": "decision_id",
                "DMNRule": "rule_id"
            }
            
            for entity_id, chunk_id in entity_chunk_map.items():
                # Try to match with each entity type
                for entity_type, id_field in id_field_map.items():
                    try:
                        result = session.run(f"""
                            MATCH (e:{entity_type} {{{id_field}: $entity_id}})
                            MATCH (c:ReferenceChunk {{chunk_id: $chunk_id}})
                            MERGE (e)-[:SUPPORTED_BY]->(c)
                            RETURN e
                        """, {"entity_id": entity_id, "chunk_id": chunk_id})
                        if result.single():
                            break
                    except:
                        continue
            
            # Role -> Skill (HAS_SKILL)
            if role_skill_map:
                for role_id, skill_ids in role_skill_map.items():
                    for skill_id in skill_ids:
                        session.run("""
                            MATCH (r:Role {role_id: $role_id})
                            MATCH (s:Skill {skill_id: $skill_id})
                            MERGE (r)-[:HAS_SKILL]->(s)
                        """, {"role_id": role_id, "skill_id": skill_id})
    
    # ==================== Query Operations ====================
    
    def get_all_processes(self) -> list[dict]:
        """Get all processes."""
        query = """
        MATCH (p:Process)
        RETURN p {.*} as process
        ORDER BY p.name
        """
        with self.session() as session:
            result = session.run(query)
            return [record["process"] for record in result]
    
    def get_process_with_details(self, proc_id: str) -> dict:
        """Get process with all related entities."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})
        OPTIONAL MATCH (p)-[:HAS_TASK]->(t:Task)
        OPTIONAL MATCH (p)-[:HAS_GATEWAY]->(g:Gateway)
        OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
        OPTIONAL MATCH (t)-[:USES_SKILL]->(s:Skill)
        RETURN p {.*} as process,
               collect(DISTINCT t {.*}) as tasks,
               collect(DISTINCT g {.*}) as gateways,
               collect(DISTINCT e {.*}) as events,
               collect(DISTINCT r {.*}) as roles,
               collect(DISTINCT s {.*}) as skills
        """
        with self.session() as session:
            result = session.run(query, {"proc_id": proc_id})
            record = result.single()
            if record:
                return {
                    "process": record["process"],
                    "tasks": record["tasks"],
                    "gateways": record["gateways"],
                    "events": record["events"],
                    "roles": record["roles"],
                    "skills": record["skills"]
                }
            return None
    
    def get_open_ambiguities(self) -> list[dict]:
        """Get all open ambiguity questions."""
        query = """
        MATCH (a:Ambiguity {status: 'open'})
        RETURN a {.*} as ambiguity
        ORDER BY a.created_at
        """
        with self.session() as session:
            result = session.run(query)
            return [record["ambiguity"] for record in result]
    
    def resolve_ambiguity(self, amb_id: str, answer: str):
        """Resolve an ambiguity with user's answer."""
        query = """
        MATCH (a:Ambiguity {amb_id: $amb_id})
        SET a.status = 'resolved',
            a.answer = $answer,
            a.resolved_at = datetime()
        RETURN a.amb_id
        """
        with self.session() as session:
            session.run(query, {"amb_id": amb_id, "answer": answer})
    
    def get_sequence_flows(self, process_id: str = None) -> list[dict]:
        """Get all NEXT relationships with their conditions.
        
        Returns:
            list of {from_id, from_type, from_name, to_id, to_type, to_name, condition}
        """
        query = """
        MATCH (from)-[r:NEXT]->(to)
        WHERE (from:Task OR from:Gateway OR from:Event)
          AND (to:Task OR to:Gateway OR to:Event)
        RETURN 
            CASE 
                WHEN from:Task THEN from.task_id
                WHEN from:Gateway THEN from.gateway_id
                WHEN from:Event THEN from.event_id
            END as from_id,
            CASE 
                WHEN from:Task THEN 'Task'
                WHEN from:Gateway THEN 'Gateway'
                WHEN from:Event THEN 'Event'
            END as from_type,
            from.name as from_name,
            CASE 
                WHEN to:Task THEN to.task_id
                WHEN to:Gateway THEN to.gateway_id
                WHEN to:Event THEN to.event_id
            END as to_id,
            CASE 
                WHEN to:Task THEN 'Task'
                WHEN to:Gateway THEN 'Gateway'
                WHEN to:Event THEN 'Event'
            END as to_type,
            to.name as to_name,
            r.condition as condition
        """
        with self.session() as session:
            result = session.run(query)
            flows = []
            for record in result:
                flows.append({
                    "from_id": record["from_id"],
                    "from_type": record["from_type"],
                    "from_name": record["from_name"],
                    "to_id": record["to_id"],
                    "to_type": record["to_type"],
                    "to_name": record["to_name"],
                    "condition": record["condition"]
                })
            return flows
    
    def search_similar_by_name(
        self, 
        entity_type: str, 
        name: str, 
        limit: int = 5
    ) -> list[dict]:
        """Full-text search for similar entities by name."""
        index_name = f"{entity_type.lower()}_name_idx"
        query = f"""
        CALL db.index.fulltext.queryNodes('{index_name}', $search_term)
        YIELD node, score
        RETURN node {{.*, score: score}} as entity
        LIMIT $limit
        """
        with self.session() as session:
            try:
                result = session.run(query, {
                    "search_term": name,
                    "limit": limit
                })
                return [record["entity"] for record in result]
            except Exception:
                return []

