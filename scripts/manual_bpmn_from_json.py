import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


NS_BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS_BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
NS_DC = "http://www.omg.org/spec/DD/20100524/DC"
NS_DI = "http://www.omg.org/spec/DD/20100524/DI"


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def main() -> None:
    ET.register_namespace("bpmn", NS_BPMN)
    ET.register_namespace("bpmndi", NS_BPMNDI)
    ET.register_namespace("dc", NS_DC)
    ET.register_namespace("di", NS_DI)

    src = Path("output/generated_procdef_from_procid.json")
    out = Path("output/generated_procdef_from_procid.bpmn")
    model = json.loads(src.read_text(encoding="utf-8"))

    roles = model.get("roles") or []
    events = model.get("events") or []
    activities = model.get("activities") or []
    sequences = model.get("sequences") or []

    process_name = model.get("processDefinitionName") or "Process"
    process_def_id = model.get("processDefinitionId") or "process_def"

    root = ET.Element(
        q(NS_BPMN, "definitions"),
        {
            "id": f"Definitions_{process_def_id}",
            "targetNamespace": "http://bpmn.io/schema/bpmn",
        },
    )
    collaboration = ET.SubElement(root, q(NS_BPMN, "collaboration"), {"id": "Collaboration_1"})
    ET.SubElement(
        collaboration,
        q(NS_BPMN, "participant"),
        {"id": "Participant_1", "name": process_name, "processRef": "Process_1"},
    )
    process = ET.SubElement(
        root,
        q(NS_BPMN, "process"),
        {"id": "Process_1", "name": process_name, "isExecutable": "true"},
    )

    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}

    for seq in sequences:
        seq_id = str(seq.get("id") or "")
        source = str(seq.get("source") or "")
        target = str(seq.get("target") or "")
        if not (seq_id and source and target):
            continue

        outgoing.setdefault(source, []).append(seq_id)
        incoming.setdefault(target, []).append(seq_id)

        attrs = {"id": seq_id, "sourceRef": source, "targetRef": target}
        condition = str(seq.get("condition") or "")
        if condition:
            attrs["name"] = condition
        ET.SubElement(process, q(NS_BPMN, "sequenceFlow"), attrs)

    start_event_id = None
    end_event_id = None

    for ev in events:
        ev_id = str(ev.get("id") or "")
        ev_type = str(ev.get("type") or "").lower()
        ev_name = str(ev.get("name") or ev_id)

        if "start" in ev_type:
            start_event_id = ev_id
            start_el = ET.SubElement(process, q(NS_BPMN, "startEvent"), {"id": ev_id, "name": ev_name})
            for seq_id in outgoing.get(ev_id, []):
                ET.SubElement(start_el, q(NS_BPMN, "outgoing")).text = seq_id
        elif "end" in ev_type:
            end_event_id = ev_id
            end_el = ET.SubElement(process, q(NS_BPMN, "endEvent"), {"id": ev_id, "name": ev_name})
            for seq_id in incoming.get(ev_id, []):
                ET.SubElement(end_el, q(NS_BPMN, "incoming")).text = seq_id

    for activity in activities:
        act_id = str(activity.get("id") or "")
        act_name = str(activity.get("name") or act_id)
        if not act_id:
            continue
        task = ET.SubElement(process, q(NS_BPMN, "userTask"), {"id": act_id, "name": act_name})
        for seq_id in incoming.get(act_id, []):
            ET.SubElement(task, q(NS_BPMN, "incoming")).text = seq_id
        for seq_id in outgoing.get(act_id, []):
            ET.SubElement(task, q(NS_BPMN, "outgoing")).text = seq_id

    if roles:
        lane_set = ET.SubElement(process, q(NS_BPMN, "laneSet"), {"id": "LaneSet_1"})
        for idx, role in enumerate(roles):
            role_name = str(role.get("name") or f"Role_{idx}")
            lane = ET.SubElement(lane_set, q(NS_BPMN, "lane"), {"id": f"Lane_{idx}", "name": role_name})

            for ev in events:
                if str(ev.get("role") or "") == role_name:
                    ev_id = str(ev.get("id") or "")
                    if ev_id:
                        ET.SubElement(lane, q(NS_BPMN, "flowNodeRef")).text = ev_id
            for activity in activities:
                if str(activity.get("role") or "") == role_name:
                    act_id = str(activity.get("id") or "")
                    if act_id:
                        ET.SubElement(lane, q(NS_BPMN, "flowNodeRef")).text = act_id

    diagram = ET.SubElement(root, q(NS_BPMNDI, "BPMNDiagram"), {"id": "BPMNDiagram_1"})
    plane = ET.SubElement(diagram, q(NS_BPMNDI, "BPMNPlane"), {"id": "BPMNPlane_1", "bpmnElement": "Collaboration_1"})

    node_order: list[str] = []
    if start_event_id:
        node_order.append(start_event_id)
    node_order.extend([str(a.get("id")) for a in activities if a.get("id")])
    if end_event_id:
        node_order.append(end_event_id)

    positions: dict[str, tuple[float, float, float, float]] = {}
    x = 120.0
    y = 140.0
    for node_id in node_order:
        is_event = node_id in {start_event_id, end_event_id}
        width = 36.0 if is_event else 120.0
        height = 36.0 if is_event else 80.0
        positions[node_id] = (x, y, width, height)

        shape = ET.SubElement(plane, q(NS_BPMNDI, "BPMNShape"), {"id": f"Shape_{node_id}", "bpmnElement": node_id})
        ET.SubElement(
            shape,
            q(NS_DC, "Bounds"),
            {"x": str(x), "y": str(y), "width": str(width), "height": str(height)},
        )
        x += 180.0

    for seq in sequences:
        seq_id = str(seq.get("id") or "")
        source = str(seq.get("source") or "")
        target = str(seq.get("target") or "")
        if source not in positions or target not in positions:
            continue

        sx, sy, sw, sh = positions[source]
        tx, ty, tw, th = positions[target]
        edge = ET.SubElement(plane, q(NS_BPMNDI, "BPMNEdge"), {"id": f"Edge_{seq_id}", "bpmnElement": seq_id})
        ET.SubElement(edge, q(NS_DI, "waypoint"), {"x": str(sx + sw), "y": str(sy + sh / 2)})
        ET.SubElement(edge, q(NS_DI, "waypoint"), {"x": str(tx), "y": str(ty + th / 2)})

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    out.write_text(pretty, encoding="utf-8")

    parsed = ET.fromstring(xml_bytes)
    ns = {"bpmn": NS_BPMN}
    print(f"out={out}")
    print(f"json_activities={len(activities)}")
    print(f"json_sequences={len(sequences)}")
    print(f"xml_userTask={len(parsed.findall('.//bpmn:userTask', ns))}")
    print(f"xml_sequenceFlow={len(parsed.findall('.//bpmn:sequenceFlow', ns))}")


if __name__ == "__main__":
    main()
