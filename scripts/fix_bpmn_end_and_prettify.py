import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


NS_BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS_BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
NS_DC = "http://www.omg.org/spec/DD/20100524/DC"
NS_DI = "http://www.omg.org/spec/DD/20100524/DI"


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def parse_bounds(shape: ET.Element) -> tuple[float, float, float, float] | None:
    b = shape.find(q(NS_DC, "Bounds"))
    if b is None:
        return None
    try:
        return (
            float(b.get("x", "0")),
            float(b.get("y", "0")),
            float(b.get("width", "0")),
            float(b.get("height", "0")),
        )
    except ValueError:
        return None


def main() -> None:
    path = Path("output/generated_procdef_from_procid.bpmn")
    tree = ET.parse(path)
    root = tree.getroot()

    ET.register_namespace("bpmn", NS_BPMN)
    ET.register_namespace("bpmndi", NS_BPMNDI)
    ET.register_namespace("dc", NS_DC)
    ET.register_namespace("di", NS_DI)
    ET.register_namespace("uengine", "http://uengine")

    process = root.find(f".//{q(NS_BPMN, 'process')}")
    if process is None:
        raise RuntimeError("bpmn:process not found")

    end_event = process.find(f"./{q(NS_BPMN, 'endEvent')}[@id='end_event']")
    if end_event is None:
        raise RuntimeError("end_event not found")

    incoming_tags = end_event.findall(q(NS_BPMN, "incoming"))
    if not incoming_tags:
        tasks = process.findall(f"./{q(NS_BPMN, 'userTask')}")
        if not tasks:
            raise RuntimeError("no userTask found")

        # Choose a terminal task: task with no outgoing.
        terminal = None
        for t in tasks:
            if not t.findall(q(NS_BPMN, "outgoing")):
                terminal = t
                break
        if terminal is None:
            terminal = tasks[-1]

        src_id = terminal.get("id", "")
        seq_id = f"SequenceFlow_{src_id}_end_event"

        # Insert new sequenceFlow near existing ones (before startEvent if possible).
        seq = ET.Element(
            q(NS_BPMN, "sequenceFlow"),
            {"id": seq_id, "sourceRef": src_id, "targetRef": "end_event"},
        )
        insert_at = len(list(process))
        children = list(process)
        for idx, c in enumerate(children):
            if c.tag in {q(NS_BPMN, "startEvent"), q(NS_BPMN, "userTask"), q(NS_BPMN, "endEvent")}:
                insert_at = idx
                break
        process.insert(insert_at, seq)

        out_tag = ET.SubElement(terminal, q(NS_BPMN, "outgoing"))
        out_tag.text = seq_id
        in_tag = ET.SubElement(end_event, q(NS_BPMN, "incoming"))
        in_tag.text = seq_id

        # Add DI edge if possible.
        plane = root.find(f".//{q(NS_BPMNDI, 'BPMNPlane')}")
        if plane is not None:
            src_shape = plane.find(f"./{q(NS_BPMNDI, 'BPMNShape')}[@bpmnElement='{src_id}']")
            tgt_shape = plane.find(f"./{q(NS_BPMNDI, 'BPMNShape')}[@bpmnElement='end_event']")
            if src_shape is not None and tgt_shape is not None:
                src_bounds = parse_bounds(src_shape)
                tgt_bounds = parse_bounds(tgt_shape)
                if src_bounds and tgt_bounds:
                    sx, sy, sw, sh = src_bounds
                    tx, ty, tw, th = tgt_bounds
                    edge = ET.SubElement(
                        plane,
                        q(NS_BPMNDI, "BPMNEdge"),
                        {"id": f"BPMNEdge_{src_id}_end_event", "bpmnElement": seq_id},
                    )
                    ET.SubElement(
                        edge,
                        q(NS_DI, "waypoint"),
                        {"x": str(sx + sw), "y": str(sy + (sh / 2.0))},
                    )
                    ET.SubElement(
                        edge,
                        q(NS_DI, "waypoint"),
                        {"x": str(tx), "y": str(ty + (th / 2.0))},
                    )

    # Pretty print
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    path.write_text(pretty, encoding="utf-8")

    # Verify
    parsed = ET.fromstring(xml_bytes)
    ns = {"bpmn": NS_BPMN}
    user_tasks = parsed.findall(".//bpmn:userTask", ns)
    seq_flows = parsed.findall(".//bpmn:sequenceFlow", ns)
    end = parsed.find(".//bpmn:endEvent[@id='end_event']", ns)
    end_in = end.findall("bpmn:incoming", ns) if end is not None else []
    print(f"path={path}")
    print(f"userTask={len(user_tasks)} sequenceFlow={len(seq_flows)} end_incoming={len(end_in)}")


if __name__ == "__main__":
    main()
