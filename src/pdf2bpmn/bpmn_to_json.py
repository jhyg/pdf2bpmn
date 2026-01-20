"""
BPMN XML을 Process Definition JSON으로 변환하는 모듈.

- `old_pdf2bpmn`의 `converters/bpmn_to_json.py`를 기반으로, 현 프로젝트 구조에 맞게 단일 모듈로 옮겼습니다.
- uEngine 확장 요소(`uengine:properties/uengine:json`)를 파싱하여 role/활동 속성(tool, instruction 등)을 복원합니다.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# BPMN 네임스페이스
BPMN_NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
    # IMPORTANT: XML 생성기/변환기에서 동일 네임스페이스를 사용해야 함
    "uengine": "http://www.uengine.org/schema/bpmn",
}


class BPMNToJSONConverter:
    """BPMN XML을 processDefinition JSON으로 변환"""

    def __init__(self):
        self.di_index: Dict[str, Dict[str, float]] = {}

    def convert(self, bpmn_xml: str, process_definition_id: str, process_name: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(bpmn_xml)
            self.di_index = self._build_di_shape_index(root)

            processes = self._extract_processes(root)
            if not processes:
                logger.warning("프로세스를 찾을 수 없습니다.")
                return self._create_empty_definition(process_definition_id, process_name)

            result: Dict[str, Any] = {
                "processDefinitionId": process_definition_id,
                "processDefinitionName": process_name,
                "data": [],
                "roles": [],
                "events": [],
                "activities": [],
                "gateways": [],
                "subProcesses": [],
                "sequences": [],
                "participants": [],
            }

            # collaboration의 participant
            participants = self._extract_participants(root, processes)
            if participants:
                result["participants"] = participants

            for process in processes:
                process_data = self._parse_process(process)
                for k in ("data", "roles", "events", "activities", "gateways", "subProcesses", "sequences"):
                    result[k].extend(process_data.get(k, []))

            result["roles"] = self._deduplicate_roles(result["roles"])
            return result
        except Exception as e:
            logger.error(f"❌ BPMN XML 변환 실패: {e}", exc_info=True)
            return self._create_empty_definition(process_definition_id, process_name)

    def _create_empty_definition(self, process_definition_id: str, process_name: str) -> Dict[str, Any]:
        return {
            "processDefinitionId": process_definition_id,
            "processDefinitionName": process_name,
            "data": [],
            "roles": [],
            "events": [],
            "activities": [],
            "gateways": [],
            "subProcesses": [],
            "sequences": [],
            "participants": [],
        }

    def _build_di_shape_index(self, root: ET.Element) -> Dict[str, Dict[str, float]]:
        """DI Shape 인덱스 생성 (위치 정보)"""
        di_index: Dict[str, Dict[str, float]] = {}
        try:
            diagrams = root.findall(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram")
            for diagram in diagrams:
                plane = diagram.find("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane")
                if plane is None:
                    continue
                shapes = plane.findall("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNShape")
                for shape in shapes:
                    bpmn_element = shape.get("bpmnElement")
                    if not bpmn_element:
                        continue
                    bounds = shape.find("{http://www.omg.org/spec/DD/20100524/DC}Bounds")
                    if bounds is None:
                        continue
                    try:
                        di_index[bpmn_element] = {
                            "x": float(bounds.get("x", 0) or 0),
                            "y": float(bounds.get("y", 0) or 0),
                            "w": float(bounds.get("width", 0) or 0),
                            "h": float(bounds.get("height", 0) or 0),
                        }
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"DI Shape 인덱스 생성 중 오류: {e}")
        return di_index

    def _extract_processes(self, root: ET.Element) -> List[ET.Element]:
        definitions = root
        # root가 definitions가 아닐 수도 있으나, 대부분 root가 definitions
        processes = definitions.findall(f".//{{{BPMN_NS['bpmn']}}}process")
        return list(processes)

    def _extract_participants(self, root: ET.Element, processes: List[ET.Element]) -> List[Dict[str, Any]]:
        participants: List[Dict[str, Any]] = []
        collabs = root.findall(f".//{{{BPMN_NS['bpmn']}}}collaboration")
        process_ids = {p.get("id") for p in processes if p.get("id")}
        for collab in collabs:
            for p in collab.findall(f".//{{{BPMN_NS['bpmn']}}}participant"):
                ref = p.get("processRef")
                if ref and ref not in process_ids:
                    continue
                participants.append(
                    {
                        "id": p.get("id", ""),
                        "name": p.get("name", ""),
                        "processRef": ref or "",
                    }
                )
        return participants

    def _parse_process(self, process: ET.Element) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "data": [],
            "roles": [],
            "events": [],
            "activities": [],
            "gateways": [],
            "subProcesses": [],
            "sequences": [],
        }

        # Lane → Role
        lanes = self._extract_lanes(process)
        result["roles"] = self._build_roles_from_lanes(lanes)
        lane_index = self._build_lane_index(lanes)

        # Variables
        result["data"] = self._extract_variables(process)

        # Events/Activities/Gateways
        result["events"] = self._extract_events(process, lane_index)
        result["activities"] = self._extract_activities(process, lane_index)
        result["gateways"] = self._extract_gateways(process, lane_index)

        # Sequences
        result["sequences"] = self._extract_sequences(process)
        return result

    def _extract_lanes(self, process: ET.Element) -> List[ET.Element]:
        lanes: List[ET.Element] = []
        lane_set = process.find(f"{{{BPMN_NS['bpmn']}}}laneSet")
        if lane_set is not None:
            lanes.extend(lane_set.findall(f"{{{BPMN_NS['bpmn']}}}lane"))
        return lanes

    def _build_lane_index(self, lanes: List[ET.Element]) -> Dict[str, str]:
        """flowNodeRef → laneName"""
        lane_index: Dict[str, str] = {}
        for lane in lanes:
            lane_name = lane.get("name", "Unknown")
            for ref in lane.findall(f"{{{BPMN_NS['bpmn']}}}flowNodeRef"):
                node_id = (ref.text or "").strip()
                if node_id:
                    lane_index[node_id] = lane_name
        return lane_index

    def _build_roles_from_lanes(self, lanes: List[ET.Element]) -> List[Dict[str, Any]]:
        roles: List[Dict[str, Any]] = []
        for lane in lanes:
            lane_name = lane.get("name", "Unknown")
            endpoint = ""
            default_endpoint = ""

            ext = lane.find(f"{{{BPMN_NS['bpmn']}}}extensionElements")
            if ext is not None:
                props = ext.find(f"{{{BPMN_NS['uengine']}}}properties")
                if props is not None:
                    props_json = props.find(f"{{{BPMN_NS['uengine']}}}json")
                    if props_json is not None and props_json.text:
                        try:
                            json_data = json.loads(props_json.text)
                            role_resolution = json_data.get("roleResolutionContext", {}) or {}
                            endpoint = role_resolution.get("endpoint", "") or ""
                            if role_resolution.get("_type") == "org.uengine.kernel.DirectRoleResolutionContext":
                                default_endpoint = endpoint
                        except json.JSONDecodeError:
                            pass

            # endpoint가 list일 수도 있음 (uEngine export 형식)
            if isinstance(endpoint, list):
                endpoint = endpoint[0] if endpoint else ""
            if isinstance(default_endpoint, list):
                default_endpoint = default_endpoint[0] if default_endpoint else ""

            roles.append(
                {
                    "name": lane_name,
                    "endpoint": endpoint if self._is_uuid(endpoint) else "",
                    "resolutionRule": None,
                    "default": default_endpoint if self._is_uuid(default_endpoint) else "",
                }
            )
        return roles

    def _is_uuid(self, value: str) -> bool:
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        return bool(uuid_pattern.match(value)) if value else False

    def _extract_variables(self, process: ET.Element) -> List[Dict[str, Any]]:
        variables: List[Dict[str, Any]] = []
        ext = process.find(f"{{{BPMN_NS['bpmn']}}}extensionElements")
        if ext is None:
            return variables
        props = ext.find(f"{{{BPMN_NS['uengine']}}}properties")
        if props is None:
            return variables
        for var_elem in props.findall(f"{{{BPMN_NS['uengine']}}}variable"):
            var_name = var_elem.get("name", "")
            var_desc = var_elem.get("description", "")
            var_type = var_elem.get("type", "string")
            if var_name:
                variables.append({"name": var_name, "description": var_desc or f"{var_name} description", "type": var_type})
        return variables

    def _extract_events(self, process: ET.Element, lane_index: Dict[str, str]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        event_types = [
            (f"{{{BPMN_NS['bpmn']}}}startEvent", "startEvent"),
            (f"{{{BPMN_NS['bpmn']}}}endEvent", "endEvent"),
            (f"{{{BPMN_NS['bpmn']}}}intermediateCatchEvent", "intermediateCatchEvent"),
            (f"{{{BPMN_NS['bpmn']}}}intermediateThrowEvent", "intermediateThrowEvent"),
        ]
        process_id = process.get("id", "")
        for tag, typ in event_types:
            for ev in process.findall(tag):
                ev_id = ev.get("id", "")
                ev_name = ev.get("name", "")
                role = lane_index.get(ev_id, "")
                events.append(
                    {
                        "id": ev_id,
                        "name": ev_name,
                        "type": typ,
                        "role": role,
                        "process": process_id,
                    }
                )
        return events

    def _extract_activities(self, process: ET.Element, lane_index: Dict[str, str]) -> List[Dict[str, Any]]:
        activities: List[Dict[str, Any]] = []
        # userTask 중심 + 일부 task 타입 대응
        task_tags = [
            (f"{{{BPMN_NS['bpmn']}}}userTask", "userTask"),
            (f"{{{BPMN_NS['bpmn']}}}serviceTask", "serviceTask"),
            (f"{{{BPMN_NS['bpmn']}}}task", "task"),
            (f"{{{BPMN_NS['bpmn']}}}manualTask", "manualTask"),
        ]

        for tag, typ in task_tags:
            for t in process.findall(tag):
                tid = t.get("id", "")
                name = t.get("name", "")
                role = lane_index.get(tid, "")

                # extension json에서 상세 필드 복원
                tool = "formHandler:defaultform"
                instruction = ""
                description = ""
                duration = 5
                input_data: List[str] = []
                output_data: List[str] = []
                checkpoints: List[str] = []

                ext = t.find(f"{{{BPMN_NS['bpmn']}}}extensionElements")
                if ext is not None:
                    props = ext.find(f"{{{BPMN_NS['uengine']}}}properties")
                    if props is not None:
                        props_json = props.find(f"{{{BPMN_NS['uengine']}}}json")
                        if props_json is not None and props_json.text:
                            try:
                                data = json.loads(props_json.text)
                                tool = data.get("tool") or tool
                                instruction = data.get("instruction") or instruction
                                description = data.get("description") or description
                                duration = data.get("duration") or duration
                                role = data.get("role") or role
                                input_data = data.get("inputData") or input_data
                                output_data = data.get("outputData") or output_data
                                checkpoints = data.get("checkpoints") or checkpoints
                            except json.JSONDecodeError:
                                pass

                activities.append(
                    {
                        "id": tid,
                        "name": name,
                        "type": typ,
                        "role": role,
                        "tool": tool,
                        "instruction": instruction,
                        "description": description,
                        "duration": duration,
                        "inputData": input_data,
                        "outputData": output_data,
                        "checkpoints": checkpoints,
                    }
                )
        return activities

    def _extract_gateways(self, process: ET.Element, lane_index: Dict[str, str]) -> List[Dict[str, Any]]:
        gateways: List[Dict[str, Any]] = []
        gateway_tags = [
            (f"{{{BPMN_NS['bpmn']}}}exclusiveGateway", "exclusiveGateway"),
            (f"{{{BPMN_NS['bpmn']}}}parallelGateway", "parallelGateway"),
            (f"{{{BPMN_NS['bpmn']}}}inclusiveGateway", "inclusiveGateway"),
        ]
        for tag, typ in gateway_tags:
            for g in process.findall(tag):
                gid = g.get("id", "")
                name = g.get("name", "")
                role = lane_index.get(gid, "")
                gateways.append(
                    {
                        "id": gid,
                        "name": name,
                        "type": typ,
                        "role": role,
                        "description": "",
                        "condition": "",
                    }
                )
        return gateways

    def _extract_sequences(self, process: ET.Element) -> List[Dict[str, Any]]:
        sequences: List[Dict[str, Any]] = []
        for s in process.findall(f"{{{BPMN_NS['bpmn']}}}sequenceFlow"):
            sid = s.get("id", "")
            name = s.get("name", "")
            source = s.get("sourceRef", "")
            target = s.get("targetRef", "")
            sequences.append({"id": sid, "name": name, "source": source, "target": target, "condition": ""})
        return sequences

    def _deduplicate_roles(self, roles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_roles = []
        for r in roles:
            key = (r.get("name") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique_roles.append(r)
        return unique_roles

