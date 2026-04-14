import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf2bpmn_agent_executor import PDF2BPMNAgentExecutor
from src.pdf2bpmn.graph.neo4j_client import Neo4jClient


PROC_ID = "e9ab95a3-8919-4808-ac91-7ddb051a9778"
OUT_PATH = "output/generated_procdef_from_procid.json"


async def main() -> None:
    ex = PDF2BPMNAgentExecutor(config={})
    neo4j = Neo4jClient()
    try:
        detail = neo4j.get_process_with_details(PROC_ID) or {}
        flows = neo4j.get_sequence_flows(PROC_ID) or []
    finally:
        neo4j.close()

    extracted = {
        "process": detail.get("process") or {},
        "tasks": detail.get("tasks") or [],
        "roles": detail.get("roles") or [],
        "gateways": detail.get("gateways") or [],
        "events": detail.get("events") or [],
        "sequence_flows": flows,
    }
    process_name = extracted["process"].get("name") or "regen-test"

    out = await ex._generate_processgpt_definition_and_bpmn(
        tenant_id="localhost",
        process_name=process_name,
        extracted=extracted,
        user_request="regen-test",
    )
    out = out or {}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    runtime = out.get("definition") or {}
    print(f"saved={OUT_PATH}")
    print(f"keys={list(out.keys())}")
    print(
        "runtime_counts=",
        {
            "activities": len(runtime.get("activities") or []),
            "events": len(runtime.get("events") or []),
            "gateways": len(runtime.get("gateways") or []),
            "sequences": len(runtime.get("sequences") or []),
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
