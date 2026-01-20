"""BPMN XML generator."""
from typing import Optional, Any
from pathlib import Path
import json
import re
from jinja2 import Template

from ..models.entities import Process, Task, Role, Gateway, Event, GatewayType, EventType


BPMN_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                  xmlns:uengine="http://www.uengine.org/schema/bpmn"
                  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
                  xmlns:di="http://www.omg.org/spec/DD/20100524/DI"
                  id="Definitions_{{ process.proc_id }}"
                  name="{{ process.name }}"
                  targetNamespace="http://bpmn.io/schema/bpmn"
                  exporter="Custom BPMN Modeler"
                  exporterVersion="1.0">

  <bpmn:collaboration id="Collaboration_{{ process.proc_id }}">
    <bpmn:participant id="Participant_{{ process.proc_id }}"
                      name="{{ process.name }}"
                      processRef="{{ process_bpmn_id }}" />
  </bpmn:collaboration>

  <bpmn:process id="{{ process_bpmn_id }}"
                name="{{ process.name }}"
                isExecutable="true"
                megaProcessId="미분류"
                majorProcessId="미분류">
    <bpmn:extensionElements>
      <uengine:properties>
        <uengine:json>{{ process_uengine_json }}</uengine:json>
      </uengine:properties>
    </bpmn:extensionElements>

    <bpmn:laneSet id="LaneSet_{{ process.proc_id }}">
      {% for role in roles %}
      <bpmn:lane id="Lane_{{ role.role_id }}" name="{{ role.name }}" resolutionRule="undefined">
        <bpmn:extensionElements>
          <uengine:properties>
            <uengine:json>{{ role_uengine_json_by_id.get(role.role_id, "{}") }}</uengine:json>
          </uengine:properties>
        </bpmn:extensionElements>
        {% for node_ref in lane_node_refs.get(role.role_id, []) %}
        <bpmn:flowNodeRef>{{ node_ref }}</bpmn:flowNodeRef>
        {% endfor %}
      </bpmn:lane>
      {% endfor %}
      {% if unassigned_tasks %}
      <bpmn:lane id="Lane_Unassigned" name="미분류" resolutionRule="undefined">
        {% for task in unassigned_tasks %}
        <bpmn:flowNodeRef>{{ task_bpmn_id_by_id.get(task.task_id) }}</bpmn:flowNodeRef>
        {% endfor %}
      </bpmn:lane>
      {% endif %}
    </bpmn:laneSet>

    <!-- Start Event -->
    <bpmn:startEvent id="{{ start_event_id }}" name="{{ start_event_name }}">
      <bpmn:extensionElements>
        <uengine:properties>
          <uengine:json>{{ start_event_uengine_json }}</uengine:json>
        </uengine:properties>
      </bpmn:extensionElements>
      {% for fid in outgoing_by_node.get(start_event_id, []) %}
      <bpmn:outgoing>{{ fid }}</bpmn:outgoing>
      {% endfor %}
    </bpmn:startEvent>

    <!-- Tasks -->
    {% for task in tasks %}
    <bpmn:userTask id="{{ task_bpmn_id_by_id.get(task.task_id) }}" name="{{ task.name }}">
      <bpmn:extensionElements>
        <uengine:properties>
          <uengine:json>{{ task_uengine_json_by_id.get(task.task_id, "{}") }}</uengine:json>
        </uengine:properties>
      </bpmn:extensionElements>
      {% for fid in incoming_by_node.get(task_bpmn_id_by_id.get(task.task_id), []) %}
      <bpmn:incoming>{{ fid }}</bpmn:incoming>
      {% endfor %}
      {% for fid in outgoing_by_node.get(task_bpmn_id_by_id.get(task.task_id), []) %}
      <bpmn:outgoing>{{ fid }}</bpmn:outgoing>
      {% endfor %}
    </bpmn:userTask>
    {% endfor %}

    <!-- Gateways -->
    {% for gateway in gateways %}
    <bpmn:exclusiveGateway id="{{ gateway_bpmn_id_by_id.get(gateway.gateway_id) }}" name="{{ gateway.description or gateway.condition or '' }}">
      <bpmn:extensionElements>
        <uengine:properties>
          <uengine:json>{{ gateway_uengine_json_by_id.get(gateway.gateway_id, "{}") }}</uengine:json>
        </uengine:properties>
      </bpmn:extensionElements>
      {% for fid in incoming_by_node.get(gateway_bpmn_id_by_id.get(gateway.gateway_id), []) %}
      <bpmn:incoming>{{ fid }}</bpmn:incoming>
      {% endfor %}
      {% for fid in outgoing_by_node.get(gateway_bpmn_id_by_id.get(gateway.gateway_id), []) %}
      <bpmn:outgoing>{{ fid }}</bpmn:outgoing>
      {% endfor %}
    </bpmn:exclusiveGateway>
    {% endfor %}

    <!-- End Event -->
    <bpmn:endEvent id="{{ end_event_id }}" name="{{ end_event_name }}">
      <bpmn:extensionElements>
        <uengine:properties>
          <uengine:json>{{ end_event_uengine_json }}</uengine:json>
        </uengine:properties>
      </bpmn:extensionElements>
      {% for fid in incoming_by_node.get(end_event_id, []) %}
      <bpmn:incoming>{{ fid }}</bpmn:incoming>
      {% endfor %}
    </bpmn:endEvent>

    <!-- Sequence Flows -->
    {% for flow in sequence_flows %}
    <bpmn:sequenceFlow id="{{ flow.id }}" sourceRef="{{ flow.source }}" targetRef="{{ flow.target }}"{% if flow.name %} name="{{ flow.name }}"{% endif %}>
      {% if flow.condition_mode_json %}
      <bpmn:extensionElements>
        <uengine:properties>
          <uengine:json>{{ flow.condition_mode_json }}</uengine:json>
        </uengine:properties>
      </bpmn:extensionElements>
      {% endif %}
    </bpmn:sequenceFlow>
    {% endfor %}

  </bpmn:process>

  <bpmndi:BPMNDiagram id="BPMNDiagram_{{ process.proc_id }}">
    <bpmndi:BPMNPlane id="BPMNPlane_{{ process.proc_id }}" bpmnElement="Collaboration_{{ process.proc_id }}">

      <bpmndi:BPMNShape id="Participant_{{ process.proc_id }}_di" bpmnElement="Participant_{{ process.proc_id }}" isHorizontal="true">
        <dc:Bounds x="0" y="0" width="{{ 1200 }}" height="{{ 120 * (roles|length if roles|length > 0 else 1) }}" />
      </bpmndi:BPMNShape>

      <!-- Lanes -->
      {% for i, role in enumerate(roles) %}
      <bpmndi:BPMNShape id="Lane_{{ role.role_id }}_di" bpmnElement="Lane_{{ role.role_id }}" isHorizontal="true">
        <dc:Bounds x="0" y="{{ i * 120 }}" width="{{ 1200 }}" height="120" />
        <bpmndi:BPMNLabel />
      </bpmndi:BPMNShape>
      {% endfor %}

      <!-- Start -->
      <bpmndi:BPMNShape id="{{ start_event_id }}_di" bpmnElement="{{ start_event_id }}">
        <dc:Bounds x="40" y="{{ 43 + start_lane_index * 120 }}" width="34" height="34" />
        <bpmndi:BPMNLabel />
      </bpmndi:BPMNShape>

      <!-- Tasks -->
      {% for i, task in enumerate(tasks) %}
      <bpmndi:BPMNShape id="{{ task_bpmn_id_by_id.get(task.task_id) }}_di" bpmnElement="{{ task_bpmn_id_by_id.get(task.task_id) }}">
        <dc:Bounds x="{{ 140 + i * 150 }}" y="{{ 20 + task_lane_index.get(task.task_id, 0) * 120 }}" width="100" height="80" />
        <bpmndi:BPMNLabel />
      </bpmndi:BPMNShape>
      {% endfor %}

      <!-- Gateways -->
      {% for i, gateway in enumerate(gateways) %}
      <bpmndi:BPMNShape id="{{ gateway_bpmn_id_by_id.get(gateway.gateway_id) }}_di" bpmnElement="{{ gateway_bpmn_id_by_id.get(gateway.gateway_id) }}" isMarkerVisible="true">
        <dc:Bounds x="{{ 140 + (tasks|length + i) * 150 }}" y="{{ 35 + gateway_lane_index.get(gateway.gateway_id, 0) * 120 }}" width="50" height="50" />
        <bpmndi:BPMNLabel />
      </bpmndi:BPMNShape>
      {% endfor %}

      <!-- End -->
      <bpmndi:BPMNShape id="{{ end_event_id }}_di" bpmnElement="{{ end_event_id }}">
        <dc:Bounds x="{{ 140 + (tasks|length + gateways|length) * 150 + 80 }}" y="{{ 43 + end_lane_index * 120 }}" width="34" height="34" />
        <bpmndi:BPMNLabel />
      </bpmndi:BPMNShape>

      <!-- Sequence Flow Edges -->
      {% for flow in sequence_flows %}
      <bpmndi:BPMNEdge id="{{ flow.id }}_di" bpmnElement="{{ flow.id }}">
        <di:waypoint x="{{ flow.source_x }}" y="{{ flow.source_y }}" />
        <di:waypoint x="{{ flow.target_x }}" y="{{ flow.target_y }}" />
        {% if flow.name %}
        <bpmndi:BPMNLabel>
          <dc:Bounds x="{{ (flow.source_x + flow.target_x) // 2 - 30 }}" y="{{ (flow.source_y + flow.target_y) // 2 - 20 }}" width="60" height="14" />
        </bpmndi:BPMNLabel>
        {% endif %}
      </bpmndi:BPMNEdge>
      {% endfor %}

    </bpmndi:BPMNPlane>
  </bpmndi:BPMNDiagram>

</bpmn:definitions>
"""


class BPMNGenerator:
    """Generate BPMN XML from extracted process data."""
    
    def __init__(self):
        self.template = Template(BPMN_TEMPLATE)
    
    def generate(
        self,
        process: Process,
        tasks: list[Task],
        roles: list[Role],
        gateways: list[Gateway],
        events: list[Event],
        task_role_map: dict[str, str] = None,
        neo4j_sequence_flows: list[dict] = None
    ) -> str:
        """Generate BPMN XML for a process.
        
        Args:
            process: The main process
            tasks: List of tasks
            roles: List of roles
            gateways: List of gateways
            events: List of events
            task_role_map: Mapping of task_id to role_id
            neo4j_sequence_flows: Sequence flows from Neo4j with conditions
                [{from_id, from_type, to_id, to_type, condition}, ...]
        """
        
        task_role_map = task_role_map or {}
        neo4j_sequence_flows = neo4j_sequence_flows or []
        
        # ----------------------------
        # uEngine-friendly IDs & JSON
        # ----------------------------
        process_bpmn_id = f"Process_{process.proc_id}"

        def _sanitize_id(text: str) -> str:
            t = (text or "").strip()
            if not t:
                return "unnamed"
            t = re.sub(r"\s+", "_", t)
            t = re.sub(r"[^0-9A-Za-z가-힣_-]", "_", t)
            return t[:60]

        def _uengine_json(obj: Any) -> str:
            # uengine:json 은 문자열 노드이므로 JSON 문자열을 그대로 넣습니다.
            return json.dumps(obj, ensure_ascii=False)

        # process extension json (minimal)
        process_uengine_json = _uengine_json(
            {
                "definitionName": process.name,
                "version": "1.0",
                "shortDescription": {"text": ""},
            }
        )

        # Map internal entities to uEngine-style BPMN element IDs
        task_bpmn_id_by_id = {t.task_id: f"Activity_{_sanitize_id(t.task_id)[:8]}" for t in tasks}
        # (task_id는 uuid라 앞 8글자 기반으로 Activity_XXXXXXXX 형태)
        task_bpmn_id_by_id = {t.task_id: f"Activity_{t.task_id[:8]}" for t in tasks}
        gateway_bpmn_id_by_id = {g.gateway_id: f"Gateway_{g.gateway_id[:8]}" for g in gateways}
        event_bpmn_id_by_id = {e.event_id: f"Event_{e.event_id[:8]}" for e in events}

        # Start/End event defaults
        start_event_id = "Event_Start"
        start_event_name = "Start"
        end_event_id = "Event_End"
        end_event_name = "End"

        start_event_uengine_json = _uengine_json({"description": "start event", "role": ""})
        end_event_uengine_json = _uengine_json({"description": "end event", "role": ""})

        # Role extension json (endpoint 정보는 현재 엔티티에 없어서 빈 값으로 둠)
        role_uengine_json_by_id = {
            r.role_id: _uengine_json(
                {
                    "roleResolutionContext": {
                        "_type": "org.uengine.kernel.DirectRoleResolutionContext",
                        "endpoint": [],
                    }
                }
            )
            for r in roles
        }

        # Task extension json (uEngine이 기대하는 필드 구조 일부)
        task_uengine_json_by_id = {}
        for t in tasks:
            rid = task_role_map.get(t.task_id)
            role_name = ""
            if rid:
                for r in roles:
                    if r.role_id == rid:
                        role_name = r.name
                        break
            task_uengine_json_by_id[t.task_id] = _uengine_json(
                {
                    "role": role_name or "",
                    "duration": 5,
                    "instruction": t.description or "",
                    "description": t.description or "",
                    "checkpoints": [],
                    "agent": None,
                    "agentMode": "none",
                    "orchestration": None,
                    "attachments": [],
                    "inputData": [],
                    "tool": "formHandler:defaultform",
                    "customProperties": [],
                }
            )

        gateway_uengine_json_by_id = {g.gateway_id: _uengine_json({"description": g.description or "", "role": ""}) for g in gateways}

        tasks_by_role = {}
        unassigned_tasks = []
        task_lane_index = {}
        gateway_lane_index = {}
        
        for i, task in enumerate(sorted(tasks, key=lambda t: t.order)):
            role_id = task_role_map.get(task.task_id)
            if role_id:
                if role_id not in tasks_by_role:
                    tasks_by_role[role_id] = []
                tasks_by_role[role_id].append(task)
                # Find lane index
                for j, role in enumerate(roles):
                    if role.role_id == role_id:
                        task_lane_index[task.task_id] = j
                        break
            else:
                unassigned_tasks.append(task)
                task_lane_index[task.task_id] = len(roles)  # Unassigned lane

        # Place gateways on first lane by default
        for g in gateways:
            gateway_lane_index[g.gateway_id] = 0
        
        # Separate start and end events
        start_events = [e for e in events if e.event_type == EventType.START]
        end_events = [e for e in events if e.event_type == EventType.END]
        
        # Build position maps for DI elements
        sorted_tasks = sorted(tasks, key=lambda t: t.order)
        element_positions = self._calculate_element_positions(
            sorted_tasks,
            gateways,
            start_events,
            end_events,
            task_lane_index,
            gateway_lane_index,
            task_bpmn_id_by_id,
            gateway_bpmn_id_by_id,
            event_bpmn_id_by_id,
            start_event_id,
            end_event_id,
        )
        
        # Generate sequence flows (using Neo4j flows if available)
        sequence_flows = self._generate_sequence_flows(
            tasks, gateways, start_events, end_events, neo4j_sequence_flows, element_positions,
            task_bpmn_id_by_id=task_bpmn_id_by_id,
            gateway_bpmn_id_by_id=gateway_bpmn_id_by_id,
            event_bpmn_id_by_id=event_bpmn_id_by_id,
            start_event_id=start_event_id,
            end_event_id=end_event_id
        )

        # Build incoming/outgoing maps for template (uEngine style)
        incoming_by_node = {}
        outgoing_by_node = {}
        for f in sequence_flows:
            outgoing_by_node.setdefault(f["source"], []).append(f["id"])
            incoming_by_node.setdefault(f["target"], []).append(f["id"])

        # Lane node refs: include start/end + tasks + gateways
        lane_node_refs = {}
        for role_id, role_tasks in tasks_by_role.items():
            lane_node_refs[role_id] = [start_event_id] + [task_bpmn_id_by_id[t.task_id] for t in role_tasks] + [end_event_id]
        # Unassigned lane is handled in template by iterating unassigned_tasks only.
        
        # Render template
        bpmn_xml = self.template.render(
            process=process,
            process_bpmn_id=process_bpmn_id,
            process_uengine_json=process_uengine_json,
            tasks=sorted(tasks, key=lambda t: t.order),
            roles=roles,
            gateways=gateways,
            events=events,
            tasks_by_role=tasks_by_role,
            unassigned_tasks=unassigned_tasks,
            start_events=start_events,
            end_events=end_events,
            sequence_flows=sequence_flows,
            task_lane_index=task_lane_index,
            gateway_lane_index=gateway_lane_index,
            process_bpmn_id_by_id=None,
            task_bpmn_id_by_id=task_bpmn_id_by_id,
            gateway_bpmn_id_by_id=gateway_bpmn_id_by_id,
            event_bpmn_id_by_id=event_bpmn_id_by_id,
            role_uengine_json_by_id=role_uengine_json_by_id,
            task_uengine_json_by_id=task_uengine_json_by_id,
            gateway_uengine_json_by_id=gateway_uengine_json_by_id,
            event_uengine_json_by_id={},
            lane_node_refs=lane_node_refs,
            incoming_by_node=incoming_by_node,
            outgoing_by_node=outgoing_by_node,
            start_event_id=start_event_id,
            start_event_name=start_event_name,
            start_event_uengine_json=start_event_uengine_json,
            start_lane_index=0,
            end_event_id=end_event_id,
            end_event_name=end_event_name,
            end_event_uengine_json=end_event_uengine_json,
            end_lane_index=0,
            enumerate=enumerate
        )
        
        return bpmn_xml
    
    def _generate_sequence_flows(
        self,
        tasks: list[Task],
        gateways: list[Gateway],
        start_events: list[Event],
        end_events: list[Event],
        neo4j_flows: list[dict] = None,
        element_positions: dict = None,
        *,
        task_bpmn_id_by_id: dict[str, str],
        gateway_bpmn_id_by_id: dict[str, str],
        event_bpmn_id_by_id: dict[str, str],
        start_event_id: str,
        end_event_id: str,
    ) -> list[dict]:
        """Generate sequence flow connections using Neo4j NEXT relationships.
        
        Args:
            neo4j_flows: [{from_id, from_type, to_id, to_type, condition}, ...]
            element_positions: {element_ref: {x, y, width, height}, ...}
        """
        flows = []
        sorted_tasks = sorted(tasks, key=lambda t: t.order)
        neo4j_flows = neo4j_flows or []
        element_positions = element_positions or {}
        
        def _flow_id(source_ref: str, target_ref: str) -> str:
            # uEngine 스타일에 맞춰 sequenceFlow id를 안정적으로 생성
            raw = f"SequenceFlow_{source_ref}_{target_ref}"
            return re.sub(r"[^0-9A-Za-z_-]", "_", raw)[:240]

        def add_flow(source_ref: str, target_ref: str, condition: str = None):
            """Helper to add flow with coordinates."""
            source_pos = element_positions.get(source_ref, {"x": 0, "y": 150, "width": 36, "height": 36})
            target_pos = element_positions.get(target_ref, {"x": 100, "y": 150, "width": 100, "height": 80})
            
            source_x, source_y, target_x, target_y = self._get_connection_points(source_pos, target_pos)
            
            fid = _flow_id(source_ref, target_ref)
            if fid in added_flow_ids:
                return fid
            flows.append({
                "id": fid,
                "source": source_ref,
                "target": target_ref,
                "name": condition if condition else None,
                # uEngine에서 조건 텍스트를 보관하기 위한 확장 속성(예시와 동일한 필드명)
                "condition_mode_json": json.dumps({"conditionMode": "text"}, ensure_ascii=False) if condition else None,
                "source_x": source_x,
                "source_y": source_y,
                "target_x": target_x,
                "target_y": target_y
            })
            return fid
        
        added_flow_ids = set()
        
        # Flow from start event to first task
        if sorted_tasks:
            first_task = sorted_tasks[0]
            target_ref = task_bpmn_id_by_id.get(first_task.task_id)
            if target_ref:
                fid = add_flow(start_event_id, target_ref)
                added_flow_ids.add(fid)
        
        # Add flows from Neo4j NEXT relationships (with conditions)
        added_pairs = set()
        for flow in neo4j_flows:
            from_id = flow.get("from_id")
            from_type = flow.get("from_type")
            to_id = flow.get("to_id")
            to_type = flow.get("to_type")
            condition = flow.get("condition")
            
            if not from_id or not to_id:
                continue
            
            def map_ref(entity_id: str, entity_type: str) -> str:
                if entity_type == "Task":
                    return task_bpmn_id_by_id.get(entity_id, "")
                if entity_type == "Gateway":
                    return gateway_bpmn_id_by_id.get(entity_id, "")
                if entity_type == "Event":
                    return event_bpmn_id_by_id.get(entity_id, "")
                return ""

            source_ref = map_ref(from_id, from_type)
            target_ref = map_ref(to_id, to_type)
            if not source_ref or not target_ref:
                continue

            fid = add_flow(source_ref, target_ref, condition)
            if fid not in added_flow_ids:
                added_flow_ids.add(fid)
            added_pairs.add((from_id, to_id))
        
        # Flows between tasks (fallback for tasks not connected via Neo4j)
        for i, task in enumerate(sorted_tasks):
            if i < len(sorted_tasks) - 1:
                next_task = sorted_tasks[i + 1]
                
                # Skip if already added from Neo4j
                if (task.task_id, next_task.task_id) in added_pairs:
                    continue

                source_ref = task_bpmn_id_by_id.get(task.task_id)
                target_ref = task_bpmn_id_by_id.get(next_task.task_id)
                if not source_ref or not target_ref:
                    continue
                fid = add_flow(source_ref, target_ref)
                added_flow_ids.add(fid)
        
        # Flow from last task to end event
        if sorted_tasks:
            last_task = sorted_tasks[-1]
            # Check if last task already connected to end event via Neo4j
            has_end_connection = any(
                f.get("from_id") == last_task.task_id and f.get("to_type") == "Event"
                for f in neo4j_flows
            )
            
            if not has_end_connection:
                source_ref = task_bpmn_id_by_id.get(last_task.task_id)
                if source_ref:
                    fid = add_flow(source_ref, end_event_id)
                    added_flow_ids.add(fid)
        
        return flows
    
    def _format_bpmn_ref(self, entity_id: str, entity_type: str) -> str:
        """(deprecated) Kept for backward compatibility."""
        return entity_id
    
    def _calculate_element_positions(
        self,
        tasks: list[Task],
        gateways: list[Gateway],
        start_events: list[Event],
        end_events: list[Event],
        task_lane_index: dict,
        gateway_lane_index: dict,
        task_bpmn_id_by_id: dict[str, str],
        gateway_bpmn_id_by_id: dict[str, str],
        event_bpmn_id_by_id: dict[str, str],
        start_event_id: str,
        end_event_id: str,
    ) -> dict:
        """Calculate x, y positions for all BPMN elements.
        
        Returns:
            dict mapping element_ref (e.g., "Task_xxx", "Gateway_xxx") to {x, y, width, height}
        """
        positions = {}
        
        # Start event position (always one: start_event_id)
        positions[start_event_id] = {"x": 40, "y": 43, "width": 34, "height": 34}
        
        # Task positions
        for i, task in enumerate(tasks):
            lane_idx = task_lane_index.get(task.task_id, 0)
            task_x = 140 + i * 150
            task_y = 20 + lane_idx * 120
            positions[task_bpmn_id_by_id.get(task.task_id, task.task_id)] = {
                "x": task_x, "y": task_y, "width": 100, "height": 80
            }
        
        # Gateway positions
        for i, gateway in enumerate(gateways):
            lane_idx = gateway_lane_index.get(gateway.gateway_id, 0)
            gw_x = 140 + (len(tasks) + i) * 150
            gw_y = 35 + lane_idx * 120
            positions[gateway_bpmn_id_by_id.get(gateway.gateway_id, gateway.gateway_id)] = {
                "x": gw_x, "y": gw_y, "width": 50, "height": 50
            }
        
        # End event position
        end_x = 140 + (len(tasks) + len(gateways)) * 150 + 80
        positions[end_event_id] = {"x": end_x, "y": 43, "width": 34, "height": 34}
        
        return positions
    
    def _get_connection_points(self, source_pos: dict, target_pos: dict) -> tuple:
        """Calculate connection points between two elements.
        
        Returns:
            tuple: (source_x, source_y, target_x, target_y)
        """
        # Source: right edge center
        source_x = source_pos["x"] + source_pos["width"]
        source_y = source_pos["y"] + source_pos["height"] // 2
        
        # Target: left edge center
        target_x = target_pos["x"]
        target_y = target_pos["y"] + target_pos["height"] // 2
        
        return source_x, source_y, target_x, target_y
    
    def save(self, bpmn_xml: str, output_path: str) -> str:
        """Save BPMN XML to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bpmn_xml, encoding="utf-8")
        return str(path)




