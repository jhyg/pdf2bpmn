"""BPMN XML generator."""
from typing import Optional
from pathlib import Path
from jinja2 import Template

from ..models.entities import Process, Task, Role, Gateway, Event, GatewayType, EventType


BPMN_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
                  xmlns:di="http://www.omg.org/spec/DD/20100524/DI"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  id="Definitions_{{ process.proc_id }}"
                  targetNamespace="http://bpmn.io/schema/bpmn">
  
  <bpmn:collaboration id="Collaboration_{{ process.proc_id }}">
    <bpmn:participant id="Participant_{{ process.proc_id }}" 
                      name="{{ process.name }}" 
                      processRef="Process_{{ process.proc_id }}" />
  </bpmn:collaboration>
  
  <bpmn:process id="Process_{{ process.proc_id }}" 
                name="{{ process.name }}" 
                isExecutable="true">
    
    <!-- Lanes for Roles -->
    <bpmn:laneSet id="LaneSet_{{ process.proc_id }}">
      {% for role in roles %}
      <bpmn:lane id="Lane_{{ role.role_id }}" name="{{ role.name }}">
        {% for task in tasks_by_role.get(role.role_id, []) %}
        <bpmn:flowNodeRef>Task_{{ task.task_id }}</bpmn:flowNodeRef>
        {% endfor %}
      </bpmn:lane>
      {% endfor %}
      {% if unassigned_tasks %}
      <bpmn:lane id="Lane_Unassigned" name="Unassigned">
        {% for task in unassigned_tasks %}
        <bpmn:flowNodeRef>Task_{{ task.task_id }}</bpmn:flowNodeRef>
        {% endfor %}
      </bpmn:lane>
      {% endif %}
    </bpmn:laneSet>
    
    <!-- Start Events -->
    {% for event in events if event.event_type.value == 'start' %}
    <bpmn:startEvent id="StartEvent_{{ event.event_id }}" name="{{ event.name }}">
      {% if event.trigger %}
      <bpmn:documentation>{{ event.trigger }}</bpmn:documentation>
      {% endif %}
      <bpmn:outgoing>Flow_Start_{{ event.event_id }}</bpmn:outgoing>
    </bpmn:startEvent>
    {% endfor %}
    
    {% if not start_events %}
    <bpmn:startEvent id="StartEvent_Default" name="Start">
      <bpmn:outgoing>Flow_Start_Default</bpmn:outgoing>
    </bpmn:startEvent>
    {% endif %}
    
    <!-- Tasks -->
    {% for task in tasks %}
    <bpmn:{% if task.task_type.value == 'human' %}userTask{% elif task.task_type.value == 'system' %}serviceTask{% else %}task{% endif %} 
         id="Task_{{ task.task_id }}" 
         name="{{ task.name }}">
      {% if task.description %}
      <bpmn:documentation>{{ task.description }}</bpmn:documentation>
      {% endif %}
      <bpmn:incoming>Flow_To_{{ task.task_id }}</bpmn:incoming>
      <bpmn:outgoing>Flow_From_{{ task.task_id }}</bpmn:outgoing>
    </bpmn:{% if task.task_type.value == 'human' %}userTask{% elif task.task_type.value == 'system' %}serviceTask{% else %}task{% endif %}>
    {% endfor %}
    
    <!-- Gateways -->
    {% for gateway in gateways %}
    <bpmn:{% if gateway.gateway_type.value == 'exclusive' %}exclusiveGateway{% elif gateway.gateway_type.value == 'parallel' %}parallelGateway{% else %}inclusiveGateway{% endif %}
         id="Gateway_{{ gateway.gateway_id }}"
         name="{{ gateway.condition or gateway.description }}">
      <bpmn:incoming>Flow_To_Gateway_{{ gateway.gateway_id }}</bpmn:incoming>
      <bpmn:outgoing>Flow_From_Gateway_{{ gateway.gateway_id }}_Yes</bpmn:outgoing>
      <bpmn:outgoing>Flow_From_Gateway_{{ gateway.gateway_id }}_No</bpmn:outgoing>
    </bpmn:{% if gateway.gateway_type.value == 'exclusive' %}exclusiveGateway{% elif gateway.gateway_type.value == 'parallel' %}parallelGateway{% else %}inclusiveGateway{% endif %}>
    {% endfor %}
    
    <!-- End Events -->
    {% for event in events if event.event_type.value == 'end' %}
    <bpmn:endEvent id="EndEvent_{{ event.event_id }}" name="{{ event.name }}">
      <bpmn:incoming>Flow_End_{{ event.event_id }}</bpmn:incoming>
    </bpmn:endEvent>
    {% endfor %}
    
    {% if not end_events %}
    <bpmn:endEvent id="EndEvent_Default" name="End">
      <bpmn:incoming>Flow_End_Default</bpmn:incoming>
    </bpmn:endEvent>
    {% endif %}
    
    <!-- Sequence Flows -->
    {% for flow in sequence_flows %}
    <bpmn:sequenceFlow id="{{ flow.id }}" 
                       sourceRef="{{ flow.source }}" 
                       targetRef="{{ flow.target }}"
                       {% if flow.name %}name="{{ flow.name }}"{% endif %}>
      {% if flow.condition %}
      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">{{ flow.condition }}</bpmn:conditionExpression>
      {% endif %}
    </bpmn:sequenceFlow>
    {% endfor %}
    
  </bpmn:process>
  
  <!-- Diagram -->
  <bpmndi:BPMNDiagram id="BPMNDiagram_{{ process.proc_id }}">
    <bpmndi:BPMNPlane id="BPMNPlane_{{ process.proc_id }}" 
                      bpmnElement="Collaboration_{{ process.proc_id }}">
      
      <!-- Participant (Pool) -->
      <bpmndi:BPMNShape id="Participant_{{ process.proc_id }}_di" 
                        bpmnElement="Participant_{{ process.proc_id }}" 
                        isHorizontal="true">
        <dc:Bounds x="160" y="80" width="{{ 300 + tasks|length * 180 }}" height="{{ 150 + (roles|length + 1) * 120 }}" />
      </bpmndi:BPMNShape>
      
      <!-- Lanes -->
      {% for i, role in enumerate(roles) %}
      <bpmndi:BPMNShape id="Lane_{{ role.role_id }}_di" 
                        bpmnElement="Lane_{{ role.role_id }}" 
                        isHorizontal="true">
        <dc:Bounds x="190" y="{{ 80 + i * 120 }}" width="{{ 270 + tasks|length * 180 }}" height="120" />
      </bpmndi:BPMNShape>
      {% endfor %}
      {% if unassigned_tasks %}
      <bpmndi:BPMNShape id="Lane_Unassigned_di" 
                        bpmnElement="Lane_Unassigned" 
                        isHorizontal="true">
        <dc:Bounds x="190" y="{{ 80 + roles|length * 120 }}" width="{{ 270 + tasks|length * 180 }}" height="120" />
      </bpmndi:BPMNShape>
      {% endif %}
      
      <!-- Start Events -->
      {% for event in events if event.event_type.value == 'start' %}
      <bpmndi:BPMNShape id="StartEvent_{{ event.event_id }}_di" bpmnElement="StartEvent_{{ event.event_id }}">
        <dc:Bounds x="232" y="{{ 130 + task_lane_index.get(tasks[0].task_id if tasks else '', 0) * 120 }}" width="36" height="36" />
      </bpmndi:BPMNShape>
      {% endfor %}
      {% if not start_events %}
      <bpmndi:BPMNShape id="StartEvent_Default_di" bpmnElement="StartEvent_Default">
        <dc:Bounds x="232" y="{{ 130 + task_lane_index.get(tasks[0].task_id if tasks else '', 0) * 120 }}" width="36" height="36" />
      </bpmndi:BPMNShape>
      {% endif %}
      
      <!-- Tasks -->
      {% for i, task in enumerate(tasks) %}
      <bpmndi:BPMNShape id="Task_{{ task.task_id }}_di" bpmnElement="Task_{{ task.task_id }}">
        <dc:Bounds x="{{ 320 + i * 180 }}" y="{{ 110 + task_lane_index.get(task.task_id, 0) * 120 }}" width="100" height="80" />
      </bpmndi:BPMNShape>
      {% endfor %}
      
      <!-- Gateways -->
      {% for i, gateway in enumerate(gateways) %}
      <bpmndi:BPMNShape id="Gateway_{{ gateway.gateway_id }}_di" bpmnElement="Gateway_{{ gateway.gateway_id }}" isMarkerVisible="true">
        <dc:Bounds x="{{ 320 + (tasks|length + i) * 180 }}" y="{{ 125 }}" width="50" height="50" />
      </bpmndi:BPMNShape>
      {% endfor %}
      
      <!-- End Events -->
      {% for event in events if event.event_type.value == 'end' %}
      <bpmndi:BPMNShape id="EndEvent_{{ event.event_id }}_di" bpmnElement="EndEvent_{{ event.event_id }}">
        <dc:Bounds x="{{ 320 + (tasks|length + gateways|length) * 180 + 50 }}" y="{{ 130 + task_lane_index.get(tasks[-1].task_id if tasks else '', 0) * 120 }}" width="36" height="36" />
      </bpmndi:BPMNShape>
      {% endfor %}
      {% if not end_events %}
      <bpmndi:BPMNShape id="EndEvent_Default_di" bpmnElement="EndEvent_Default">
        <dc:Bounds x="{{ 320 + (tasks|length + gateways|length) * 180 + 50 }}" y="{{ 130 + task_lane_index.get(tasks[-1].task_id if tasks else '', 0) * 120 }}" width="36" height="36" />
      </bpmndi:BPMNShape>
      {% endif %}
      
      <!-- Sequence Flow Edges -->
      {% for flow in sequence_flows %}
      <bpmndi:BPMNEdge id="{{ flow.id }}_di" bpmnElement="{{ flow.id }}">
        <di:waypoint x="{{ flow.source_x }}" y="{{ flow.source_y }}" />
        <di:waypoint x="{{ flow.target_x }}" y="{{ flow.target_y }}" />
        {% if flow.condition %}
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
        
        # Organize tasks by role
        tasks_by_role = {}
        unassigned_tasks = []
        task_lane_index = {}
        
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
        
        # Separate start and end events
        start_events = [e for e in events if e.event_type == EventType.START]
        end_events = [e for e in events if e.event_type == EventType.END]
        
        # Build position maps for DI elements
        sorted_tasks = sorted(tasks, key=lambda t: t.order)
        element_positions = self._calculate_element_positions(
            sorted_tasks, gateways, start_events, end_events, task_lane_index, len(roles)
        )
        
        # Generate sequence flows (using Neo4j flows if available)
        sequence_flows = self._generate_sequence_flows(
            tasks, gateways, start_events, end_events, neo4j_sequence_flows, element_positions
        )
        
        # Render template
        bpmn_xml = self.template.render(
            process=process,
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
        element_positions: dict = None
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
        
        def add_flow(flow_id: str, source_ref: str, target_ref: str, condition: str = None):
            """Helper to add flow with coordinates."""
            source_pos = element_positions.get(source_ref, {"x": 0, "y": 150, "width": 36, "height": 36})
            target_pos = element_positions.get(target_ref, {"x": 100, "y": 150, "width": 100, "height": 80})
            
            source_x, source_y, target_x, target_y = self._get_connection_points(source_pos, target_pos)
            
            flows.append({
                "id": flow_id,
                "source": source_ref,
                "target": target_ref,
                "name": condition if condition else None,
                "condition": condition,
                "source_x": source_x,
                "source_y": source_y,
                "target_x": target_x,
                "target_y": target_y
            })
        
        added_flow_ids = set()
        
        # Flow from start event to first task
        if sorted_tasks:
            first_task = sorted_tasks[0]
            if start_events:
                for event in start_events:
                    flow_id = f"Flow_Start_{event.event_id}"
                    if flow_id not in added_flow_ids:
                        add_flow(flow_id, f"StartEvent_{event.event_id}", f"Task_{first_task.task_id}")
                        added_flow_ids.add(flow_id)
            else:
                flow_id = "Flow_Start_Default"
                if flow_id not in added_flow_ids:
                    add_flow(flow_id, "StartEvent_Default", f"Task_{first_task.task_id}")
                    added_flow_ids.add(flow_id)
        
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
            
            # Format source and target refs based on type
            source_ref = self._format_bpmn_ref(from_id, from_type)
            target_ref = self._format_bpmn_ref(to_id, to_type)
            
            flow_id = f"Flow_{from_id[:8]}_{to_id[:8]}"
            if flow_id not in added_flow_ids:
                add_flow(flow_id, source_ref, target_ref, condition)
                added_flow_ids.add(flow_id)
                added_pairs.add((from_id, to_id))
        
        # Flows between tasks (fallback for tasks not connected via Neo4j)
        for i, task in enumerate(sorted_tasks):
            if i < len(sorted_tasks) - 1:
                next_task = sorted_tasks[i + 1]
                
                # Skip if already added from Neo4j
                if (task.task_id, next_task.task_id) in added_pairs:
                    continue
                
                flow_id = f"Flow_{task.task_id[:8]}_{next_task.task_id[:8]}"
                if flow_id not in added_flow_ids:
                    add_flow(flow_id, f"Task_{task.task_id}", f"Task_{next_task.task_id}")
                    added_flow_ids.add(flow_id)
        
        # Flow from last task to end event
        if sorted_tasks:
            last_task = sorted_tasks[-1]
            # Check if last task already connected to end event via Neo4j
            has_end_connection = any(
                f.get("from_id") == last_task.task_id and f.get("to_type") == "Event"
                for f in neo4j_flows
            )
            
            if not has_end_connection:
                if end_events:
                    for event in end_events:
                        flow_id = f"Flow_End_{event.event_id}"
                        if flow_id not in added_flow_ids:
                            add_flow(flow_id, f"Task_{last_task.task_id}", f"EndEvent_{event.event_id}")
                            added_flow_ids.add(flow_id)
                else:
                    flow_id = "Flow_End_Default"
                    if flow_id not in added_flow_ids:
                        add_flow(flow_id, f"Task_{last_task.task_id}", "EndEvent_Default")
                        added_flow_ids.add(flow_id)
        
        return flows
    
    def _format_bpmn_ref(self, entity_id: str, entity_type: str) -> str:
        """Format BPMN element reference based on entity type."""
        if entity_type == "Task":
            return f"Task_{entity_id}"
        elif entity_type == "Gateway":
            return f"Gateway_{entity_id}"
        elif entity_type == "Event":
            return f"Event_{entity_id}"
        else:
            return entity_id
    
    def _calculate_element_positions(
        self,
        tasks: list[Task],
        gateways: list[Gateway],
        start_events: list[Event],
        end_events: list[Event],
        task_lane_index: dict,
        num_roles: int
    ) -> dict:
        """Calculate x, y positions for all BPMN elements.
        
        Returns:
            dict mapping element_ref (e.g., "Task_xxx", "Gateway_xxx") to {x, y, width, height}
        """
        positions = {}
        
        # Start event position
        start_x = 250
        start_y = 148  # Center of first lane
        
        if start_events:
            for event in start_events:
                positions[f"StartEvent_{event.event_id}"] = {
                    "x": start_x, "y": start_y, "width": 36, "height": 36
                }
        else:
            positions["StartEvent_Default"] = {
                "x": start_x, "y": start_y, "width": 36, "height": 36
            }
        
        # Task positions
        for i, task in enumerate(tasks):
            lane_idx = task_lane_index.get(task.task_id, 0)
            task_x = 320 + i * 180
            task_y = 110 + lane_idx * 120
            positions[f"Task_{task.task_id}"] = {
                "x": task_x, "y": task_y, "width": 100, "height": 80
            }
        
        # Gateway positions
        for i, gateway in enumerate(gateways):
            gw_x = 320 + (len(tasks) + i) * 180
            gw_y = 125
            positions[f"Gateway_{gateway.gateway_id}"] = {
                "x": gw_x, "y": gw_y, "width": 50, "height": 50
            }
        
        # End event position
        end_x = 320 + (len(tasks) + len(gateways)) * 180 + 50
        end_y = 148
        
        if end_events:
            for event in end_events:
                positions[f"EndEvent_{event.event_id}"] = {
                    "x": end_x, "y": end_y, "width": 36, "height": 36
                }
        else:
            positions["EndEvent_Default"] = {
                "x": end_x, "y": end_y, "width": 36, "height": 36
            }
        
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




