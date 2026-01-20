#!/usr/bin/env python3
"""
PDF2BPMN AgentExecutor for ProcessGPT SDK
ProcessGPT SDK의 AgentExecutor 인터페이스를 구현한 PDF2BPMN 에이전트
PDF를 분석하여 BPMN XML을 생성하고, 진행 상황을 실시간으로 이벤트로 전송
"""

import asyncio
import os
import logging
import uuid
import json
import re
import httpx
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import traceback
import xml.etree.ElementTree as ET
import sys
from html.parser import HTMLParser

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Supabase imports
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase-py not available. Install with: pip install supabase")

# ProcessGPT SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.types import TaskStatusUpdateEvent, TaskState, TaskArtifactUpdateEvent
    from a2a.utils import new_agent_text_message, new_text_artifact
    PROCESSGPT_SDK_AVAILABLE = True
except ImportError:
    # Fallback classes for when SDK is not available
    class AgentExecutor:
        async def execute(self, context, event_queue): pass
        async def cancel(self, context, event_queue): pass
    
    class RequestContext:
        def get_user_input(self): return ""
        def get_context_data(self): return {}
    
    class EventQueue:
        def enqueue_event(self, event): pass
    
    class TaskStatusUpdateEvent:
        def __init__(self, **kwargs): pass
    
    class TaskState:
        working = "working"
        input_required = "input_required"
    
    class TaskArtifactUpdateEvent:
        def __init__(self, **kwargs): pass
    
    def new_agent_text_message(text, context_id, task_id): return text
    def new_text_artifact(name, description, text): return {"name": name, "description": description, "text": text}
    
    PROCESSGPT_SDK_AVAILABLE = False
    print("Warning: ProcessGPT SDK not available. Using fallback classes.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Form generation prompt data (ported from process-gpt-vue3)
# - Source: process-gpt-vue3/src/components/ai/FormDesignGeneratorPromptSnipptsData.js
# - NOTE: examples are intentionally omitted to reduce token/cost; rules + component specs are kept.
# ---------------------------------------------------------------------------

FORM_CONTAINER_SPACE_SETS: List[List[int]] = [
    [12],
    [6, 6],
    [4, 8],
    [8, 4],
    [4, 4, 4],
    [3, 6, 3],
    [3, 3, 3, 3],
]

FORM_COMPONENT_INFOS: List[Dict[str, str]] = [
    {
        "tagName": "boolean-field",
        "tag": "<boolean-field name='<unique_identifier>' alias='<display_label>' disabled='<true|false>' readonly='<true|false>'></boolean-field>",
        "purpose": "To select either 'true' or 'false'",
        "limit": "",
    },
    {
        "tagName": "user-select-field",
        "tag": "<user-select-field name='<unique_identifier>' alias='<display_label>' disabled='<true|false>' readonly='<true|false>'></user-select-field>",
        "purpose": "To select users from the system",
        "limit": "",
    },
    {
        "tagName": "select-field",
        "tag": (
            "<select-field name='<unique_identifier>' alias='<display_label>' is_dynamic_load='<fixed|urlBinding>' "
            "items='<options_list_when_is_dynamic_load_is_false>' "
            "dynamic_load_url='<JSON_data_load_URL_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_key_json_path='<JSON_PATH_for_key_array_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_value_json_path='<JSON_PATH_for_value_array_when_is_dynamic_load_is_urlBinding>' "
            "disabled='<true|false>' readonly='<true|false>'></select-field>"
        ),
        "purpose": "To select one option from multiple choices",
        "limit": (
            "When is_dynamic_load is fixed, items is required and must be formatted as "
            """'[{"key1": "label1"}, {"key2": "label2"}]'. """
            "When is_dynamic_load is urlBinding, dynamic_load_url, dynamic_load_key_json_path, and "
            "dynamic_load_value_json_path are all required."
        ),
    },
    {
        "tagName": "checkbox-field",
        "tag": (
            "<checkbox-field name='<unique_identifier>' alias='<display_label>' is_dynamic_load='<fixed|urlBinding>' "
            "items='<options_list_when_is_dynamic_load_is_false>' "
            "dynamic_load_url='<JSON_data_load_URL_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_key_json_path='<JSON_PATH_for_key_array_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_value_json_path='<JSON_PATH_for_value_array_when_is_dynamic_load_is_urlBinding>' "
            "disabled='<true|false>' readonly='<true|false>'></checkbox-field>"
        ),
        "purpose": "To select multiple options from a list of choices",
        "limit": (
            "When is_dynamic_load is fixed, items is required and must be formatted as "
            """'[{"key1": "label1"}, {"key2": "label2"}]'. """
            "When is_dynamic_load is urlBinding, dynamic_load_url, dynamic_load_key_json_path, and "
            "dynamic_load_value_json_path are all required."
        ),
    },
    {
        "tagName": "radio-field",
        "tag": (
            "<radio-field name='<unique_identifier>' alias='<display_label>' is_dynamic_load='<fixed|urlBinding>' "
            "items='<options_list_when_is_dynamic_load_is_false>' "
            "dynamic_load_url='<JSON_data_load_URL_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_key_json_path='<JSON_PATH_for_key_array_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_value_json_path='<JSON_PATH_for_value_array_when_is_dynamic_load_is_urlBinding>' "
            "disabled='<true|false>' readonly='<true|false>'></radio-field>"
        ),
        "purpose": "To select one option from multiple listed choices (displayed as radio buttons)",
        "limit": (
            "When is_dynamic_load is fixed, items is required and must be formatted as "
            """'[{"key1": "label1"}, {"key2": "label2"}]'. """
            "When is_dynamic_load is urlBinding, dynamic_load_url, dynamic_load_key_json_path, and "
            "dynamic_load_value_json_path are all required."
        ),
    },
    {
        "tagName": "file-field",
        "tag": "<file-field name='<unique_identifier>' alias='<display_label>' disabled='<true|false>' readonly='<true|false>'></file-field>",
        "purpose": "To upload files",
        "limit": "",
    },
    {
        "tagName": "label-field",
        "tag": "<label-field label='<label_text>'></label-field>",
        "purpose": "To provide descriptive text for components",
        "limit": "Not needed for components that already have name and alias attributes (which automatically generate labels)",
    },
    {
        "tagName": "report-field",
        "tag": "<report-field name='<unique_identifier>' alias='<display_label>'></report-field>",
        "purpose": "To collect markdown input",
        "limit": "Write markdown body only; use '---' as section separators when needed.",
    },
    {
        "tagName": "slide-field",
        "tag": "<slide-field name='<unique_identifier>' alias='<display_label>'></slide-field>",
        "purpose": "To collect slide input",
        "limit": "Write markdown body only; use '---' as section separators when needed.",
    },
    {
        "tagName": "bpmn-uengine-field",
        "tag": "<bpmn-uengine-field name='<unique_identifier>' alias='<display_label>'></bpmn-uengine-field>",
        "purpose": "To collect BPMN process definitions as XML",
        "limit": "Use this field when the user explicitly asks for a BPMN process editor or diagram input.",
    },
    {
        "tagName": "text-field",
        "tag": "<text-field name='<unique_identifier>' alias='<display_label>' type='<text|number|email|url|date|datetime-local|month|week|time|password|tel|color>' disabled='<true|false>' readonly='<true|false>'></text-field>",
        "purpose": "To collect various types of text input",
        "limit": "For selections with many options (like years), use text-field instead of select-field",
    },
    {
        "tagName": "textarea-field",
        "tag": "<textarea-field name='<unique_identifier>' alias='<display_label>' rows='<number_of_rows>' disabled='<true|false>' readonly='<true|false>'></textarea-field>",
        "purpose": "To collect multi-line text input",
        "limit": "",
    },
]


class PDF2BPMNAgentExecutor(AgentExecutor):
    """
    ProcessGPT SDK와 호환되는 PDF2BPMN AgentExecutor
    PDF 파일을 분석하여 BPMN XML을 생성하는 에이전트
    
    지원 기능:
    - PDF URL 다운로드 및 분석
    - 다중 프로세스 BPMN 생성
    - 실시간 진행 상황 이벤트 발송
    - proc_def, configuration(proc_map) 저장
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        PDF2BPMN AgentExecutor 초기화
        
        Args:
            config: 설정 딕셔너리
                - pdf2bpmn_url: PDF2BPMN 서버 URL (기본: http://localhost:8001)
                - timeout: API 호출 타임아웃 (초)
                - supabase_url: Supabase URL
                - supabase_key: Supabase 서비스 키
        """
        self.config = config or {}
        self.is_cancelled = False
        
        # PDF2BPMN 서버 설정
        self.pdf2bpmn_url = os.getenv('PDF2BPMN_URL', self.config.get('pdf2bpmn_url', 'http://localhost:8001'))
        self.timeout = self.config.get('timeout', 3600)  # 1시간 타임아웃
        
        # Supabase 설정
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SERVICE_ROLE_KEY')
        self.supabase_client: Optional[Client] = None

        # OpenAI client (for form generation)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.openai_client: Optional[OpenAI] = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                logger.warning(f"[WARN] OpenAI client init failed: {e}")
                self.openai_client = None
        
        # HTTP 클라이언트
        self.http_client: Optional[httpx.AsyncClient] = None

        # Org/agent cache (lazy)
        self._org_loaded: bool = False
        self._org_teams_by_name: Dict[str, str] = {}  # normalized team name -> team id
        self._agents: List[Dict[str, Any]] = []  # users table agents
        
        # Supabase 초기화
        self._setup_supabase()
        
        logger.info(f"[OK] PDF2BPMNAgentExecutor initialized")
        logger.info(f"    - PDF2BPMN Server: {self.pdf2bpmn_url}")
        logger.info(f"    - Timeout: {self.timeout}s")

    def _setup_supabase(self):
        """Supabase 클라이언트 초기화"""
        if not SUPABASE_AVAILABLE:
            logger.warning("[WARN] Supabase library not installed.")
            return
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("[WARN] Supabase URL or key not configured.")
            return
        
        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"[OK] Supabase client initialized")
        except Exception as e:
            logger.error(f"[ERROR] Supabase client init failed: {e}")
            self.supabase_client = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 반환 (lazy initialization)"""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=self.timeout)
        return self.http_client

    # -----------------------------------------------------------------------
    # Form generation + saving (B안: proc_def 저장 후 폼 생성/저장)
    # -----------------------------------------------------------------------

    def _build_form_generator_base_messages(self) -> List[Dict[str, Any]]:
        """FormDesignGenerator.js의 시스템/가이드 프롬프트를 python용으로 구성합니다."""
        container_space_sets_prompt_str = ", ".join("{" + ", ".join(map(str, s)) + "}" for s in FORM_CONTAINER_SPACE_SETS)

        component_infos_prompt_str = "\n".join(
            [
                "#### {tagName}\n"
                "1. Tag Syntax\n"
                "`{tag}`\n\n"
                "2. Purpose\n"
                "{purpose}{limit_part}\n".format(
                    tagName=c["tagName"],
                    tag=c["tag"],
                    purpose=c.get("purpose", ""),
                    limit_part=("\n\n3. Limitation\n" + c["limit"]) if c.get("limit") else "",
                )
                for c in FORM_COMPONENT_INFOS
            ]
        )

        # NOTE:
        # - datasourcePrompt/datasourceURL은 워커 환경에서 보통 없음 → null로 두고 사용 금지 가이드만 둠
        datasource_prompt = "null"
        datasource_url = "null"

        system = {
            "role": "system",
            "content": (
                "# Role\n"
                "You are an HTML form creator assistant for process management systems, designed to generate and modify structured forms with precision and adherence to specific component guidelines.\n\n"
                "## Expertise\n"
                "- Expert in creating semantically structured HTML forms for business process management\n"
                "- Proficient in implementing grid-based layouts with proper containment hierarchies\n"
                "- Skilled at translating user requirements into functional forms\n"
                "- Specialized in component organization and responsive column distribution\n\n"
                "## Behavior Guidelines\n"
                "- Generate forms that strictly adhere to the provided component specifications\n"
                "- Maintain consistency in naming patterns and attribute formats\n"
                "- Produce clean, well-structured HTML that follows established patterns\n"
                "- Verify uniqueness of all name attributes across the entire form\n\n"
                "## Output Standards\n"
                "- Provide only valid HTML that conforms to the specified tag structure\n"
                "- Return responses in the exact JSON format specified in the guidelines\n\n"
                "# Instruction for DataSource Use\n"
                "You may be given a set of available dataSources before generating fields.\n"
                "If there is no datasource or datasourceURL is null, do not use dataSources.\n"
            ),
        }

        user_guideline = {
            "role": "user",
            "content": (
                "# Task Guidelines\n"
                "## About Task\n"
                "You create forms based on user instructions.\n"
                "You must only use the tags specified in the provided documentation.\n\n"
                "## Creating a Form from Scratch\n"
                "### Layout Structure\n"
                "First, create a layout to contain components.\n\n"
                "Layout example:\n"
                "```html\n"
                "<section>\n"
                "  <div class='row' name='<unique_layout_name>' alias='<layout_display_name>' is_multidata_mode='<true|false>'>\n"
                "      <div class='col-sm-6'>\n"
                "      </div>\n"
                "      <div class='col-sm-6'>\n"
                "      </div>\n"
                "  </div>\n"
                "</section>\n"
                "```\n\n"
                "- A section must contain exactly one div with class='row'.\n"
                "- Inside a div with class='row', you must include divs with class='col-sm-{number}'.\n"
                "- The sum of all {number} values in a row must equal 12.\n"
                f"- You must use one of these column combinations: [{container_space_sets_prompt_str}]\n"
                "- Layouts can be nested by placing a new section inside a col-sm div.\n\n"
                "### Adding Components\n"
                "- All components must be placed inside a div with class='col-sm-{number}'.\n"
                "- Every name attribute (including in div.row) must be unique.\n"
                "- For non-array string attributes, only use Korean characters, numbers, English letters, spaces, underscores(_), hyphens(-), and periods(.)\n"
                "- When creating a form, if there is no suitable result to create (insufficient task information), a text area with a default label of \"Free Input\" should be created. The form must exist.\n\n"
                "### How to infer fields from task information (flexible)\n"
                "- Use the task name/description/instruction to infer the minimum necessary inputs.\n"
                "- Prefer concrete business fields (dates, amounts, identifiers, decision/result, comment, attachments) when the text suggests them.\n"
                "- If the task clearly involves a human decision (e.g., approval/reject/hold), include fields for decision and rationale.\n"
                "- If the task involves money/payment/deposit, include date/amount/payer/proof fields.\n"
                "- If the task involves review/verification, include result and comment fields.\n"
                "- If the task involves contract/signature, include contract id/date/sign method fields.\n"
                "- These are suggestions: do NOT invent details that contradict the document; when uncertain, fall back to Free Input.\n\n"
                "### Available components\n"
                f"{component_infos_prompt_str}\n\n"
                f"{datasource_prompt}\n"
                "# Datasource URL\n"
                f"{datasource_url}\n\n"
                "### Output Format\n"
                "When responding, provide only the JSON response in markdown format, wrapped in triple backticks:\n"
                "```json\n"
                "{\n"
                '  "htmlOutput": "Generated form HTML code"\n'
                "}\n"
                "```\n"
            ),
        }

        assistant_ack = {"role": "assistant", "content": "Approved."}
        return [system, user_guideline, assistant_ack]

    def _make_fallback_form_html(self) -> str:
        # 프롬프트 가이드(폼은 비어있으면 안 됨)에 맞춘 안전한 최소 폼
        return (
            "<section>"
            "  <div class='row' name='free_input_layout' alias='Free Input' is_multidata_mode='false'>"
            "    <div class='col-sm-12'>"
            "      <textarea-field name='free_input' alias='Free Input' rows='5' disabled='false' readonly='false'></textarea-field>"
            "    </div>"
            "  </div>"
            "</section>"
        )

    async def _call_openai_for_form_html(self, request_text: str) -> str:
        """LLM 호출로 폼 HTML 생성. 실패 시 예외를 던집니다(상위에서 폴백 처리)."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client is not configured (missing OPENAI_API_KEY or openai package).")

        messages = self._build_form_generator_base_messages()
        # FormDesignGenerator의 noteMessage와 유사: alias는 한국어, name은 영어 권장
        note = "Please write values such as alias and label of the form being created in Korean. However, make sure all name attributes are written in English only."
        user_message = (
            "# Request Type\n"
            "Create\n\n"
            "# Request\n"
            f"{request_text}\n\n"
            "# Note\n"
            f"{note}\n"
        )
        messages.append({"role": "user", "content": user_message})

        def _run():
            return self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=float(os.getenv("FORM_LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("FORM_LLM_MAX_TOKENS", "2500")),
            )

        resp = await asyncio.to_thread(_run)
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Empty LLM response.")

        # 응답이 ```json ... ``` 형태일 수 있음 → code fence 제거
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
        if fence_match:
            content = fence_match.group(1).strip()

        try:
            obj = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM JSON: {e}. raw={content[:300]}...")

        html = (obj.get("htmlOutput") or "").strip()
        if not html:
            raise RuntimeError("LLM JSON did not include htmlOutput.")
        return html

    def _extract_fields_json_from_form_html(self, html: str) -> List[Dict[str, Any]]:
        """프론트 `extractFields()` 로직을 python으로 포팅."""

        field_tags = {
            "text-field",
            "select-field",
            "checkbox-field",
            "radio-field",
            "file-field",
            "label-field",
            "boolean-field",
            "textarea-field",
            "user-select-field",
            "report-field",
            "slide-field",
            "bpmn-uengine-field",
        }

        class _FieldParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.fields: List[Dict[str, Any]] = []

            def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]):
                t = (tag or "").lower()
                if t not in field_tags:
                    return
                attr = {k.lower(): v for (k, v) in attrs if k}

                alias = attr.get("alias") or ""
                name_attr = attr.get("name") or ""
                v_model = attr.get("v-model") or ""

                # v-model 바인딩에서 bracket 표기법 키 우선 추출, 없으면 name 사용
                key = name_attr
                m = re.search(r"\[['\"](.+?)['\"]\]", v_model)
                if m and m.group(1):
                    key = m.group(1)

                field_type = attr.get("type") or t.replace("-field", "")
                disabled = attr.get("disabled") if "disabled" in attr else False
                readonly = attr.get("readonly") if "readonly" in attr else False

                self.fields.append(
                    {
                        "text": alias,
                        "key": key,
                        "type": field_type,
                        "disabled": disabled,
                        "readonly": readonly,
                    }
                )

        parser = _FieldParser()
        parser.feed(html or "")
        return parser.fields

    async def _save_form_def(self, *, form_def: Dict[str, Any], tenant_id: str) -> bool:
        """form_def 테이블에 저장 (프론트 putRawDefinition(type=form)과 호환되는 컬럼 사용)."""
        if not self.supabase_client:
            logger.error("[ERROR] Supabase client is None! Cannot save form_def")
            return False

        try:
            proc_def_id = form_def.get("proc_def_id")
            activity_id = form_def.get("activity_id")
            form_id = form_def.get("id")

            if not proc_def_id or not activity_id or not form_id:
                raise ValueError("form_def requires id/proc_def_id/activity_id")

            # 기존 row 탐색(프론트와 동일 기준: tenant_id + proc_def_id + activity_id)
            existing = (
                self.supabase_client.table("form_def")
                .select("uuid,id")
                .eq("tenant_id", tenant_id)
                .eq("proc_def_id", proc_def_id)
                .eq("activity_id", activity_id)
                .execute()
            )

            if existing.data and len(existing.data) > 0:
                existing_uuid = existing.data[0].get("uuid")
                # uuid가 있으면 uuid 기준 업데이트(레거시 호환)
                if existing_uuid:
                    self.supabase_client.table("form_def").update(
                        {
                            "id": form_id,
                            "html": form_def.get("html"),
                            "proc_def_id": proc_def_id,
                            "activity_id": activity_id,
                            "fields_json": form_def.get("fields_json") or [],
                            "tenant_id": tenant_id,
                        }
                    ).eq("uuid", existing_uuid).execute()
                else:
                    # uuid가 없으면 id 기준으로 업데이트 시도
                    self.supabase_client.table("form_def").update(
                        {
                            "html": form_def.get("html"),
                            "fields_json": form_def.get("fields_json") or [],
                        }
                    ).eq("id", form_id).execute()
            else:
                self.supabase_client.table("form_def").insert(
                    {
                        "id": form_id,
                        "html": form_def.get("html"),
                        "proc_def_id": proc_def_id,
                        "activity_id": activity_id,
                        "fields_json": form_def.get("fields_json") or [],
                        "tenant_id": tenant_id,
                    }
                ).execute()

            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to save form_def: {e}")
            logger.error(traceback.format_exc())
            return False

    def _compute_form_def_id(self, *, proc_def_id: str, activity: Dict[str, Any]) -> str:
        """프론트와 동일한 form id 결정 규칙."""
        tool = (activity.get("tool") or "").strip()
        activity_id = (activity.get("id") or "").strip()

        form_id = ""
        if tool.startswith("formHandler:"):
            form_id = tool.replace("formHandler:", "", 1).strip()
        if not form_id:
            form_id = f"{proc_def_id}_{activity_id}_form"

        # 프론트는 '/'를 '#'로 치환
        form_id = form_id.replace("/", "#")
        if not form_id or form_id == "defaultform":
            form_id = f"{proc_def_id}_{activity_id.lower()}_form"
        return form_id

    async def _ensure_forms_for_process(
        self,
        *,
        proc_def_id: str,
        process_name: str,
        proc_json: Dict[str, Any],
        tenant_id: str,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """proc_def 저장 후, activity별 폼 생성+저장을 완료합니다(프론트가 없어도 수행)."""
        activities = proc_json.get("activities") or []
        if not isinstance(activities, list) or not activities:
            return {"forms_saved": 0, "activities": 0}

        forms_saved = 0
        total = len(activities)
        max_forms = int(os.getenv("FORM_MAX_PER_PROCESS", "200"))
        if total > max_forms:
            activities = activities[:max_forms]
            total = len(activities)

        for idx, a in enumerate(activities):
            if not isinstance(a, dict):
                continue

            activity_id = str(a.get("id") or f"Activity_{idx+1}")
            activity_name = str(a.get("name") or f"활동 {idx+1}")
            role_name = str(a.get("role") or "")
            instruction = str(a.get("instruction") or "")
            description = str(a.get("description") or "")
            input_data = a.get("inputData") or []
            output_data = a.get("outputData") or []

            form_def_id = self._compute_form_def_id(proc_def_id=proc_def_id, activity=a)
            # IMPORTANT:
            # - form_def는 우리가 생성/저장하므로, 프로세스 정의(activity.tool)도 동일 id를 참조해야 프론트가 기본폼(defaultform) 대신 생성된 폼을 사용합니다.
            # - proc_def는 이미 저장되었더라도, 상위에서 definition 업데이트를 다시 수행합니다.
            a["tool"] = f"formHandler:{form_def_id}"

            await self._send_progress_event(
                event_queue,
                context_id,
                task_id,
                job_id,
                f"[FORM] 폼 생성 시작 ({idx+1}/{total}): {process_name} / {activity_name}",
                "tool_usage_started",
                92,
                {"proc_def_id": proc_def_id, "activity_id": activity_id, "form_def_id": form_def_id},
            )

            request_text = (
                f"다음 BPM 프로세스의 사용자 태스크에 필요한 입력 폼을 생성하세요.\n\n"
                f"- 프로세스명: {process_name}\n"
                f"- 프로세스ID(proc_def_id): {proc_def_id}\n"
                f"- 태스크ID(activity_id): {activity_id}\n"
                f"- 태스크명: {activity_name}\n"
                f"- 담당 역할: {role_name}\n\n"
                f"태스크 설명:\n{description}\n\n"
                f"태스크 지시사항(instruction):\n{instruction}\n\n"
                f"입력 데이터 후보(inputData): {json.dumps(input_data, ensure_ascii=False)}\n"
                f"출력 데이터 후보(outputData): {json.dumps(output_data, ensure_ascii=False)}\n\n"
                f"요구사항:\n"
                f"- 태스크 수행에 필요한 최소 입력 필드를 포함하세요.\n"
                f"- 필드 alias는 한국어로, name은 영어로 작성하세요.\n"
                f"- 태스크 정보가 충분하지 않다면, 자유입력(Free Input) 중심의 폼이 생성되어도 괜찮습니다.\n"
            )

            html = ""
            # 1) LLM 시도
            try:
                per_form_timeout = float(os.getenv("FORM_LLM_TIMEOUT_SEC", "120"))
                html = await asyncio.wait_for(self._call_openai_for_form_html(request_text), timeout=per_form_timeout)
            except Exception as e:
                # 운영상 폼은 반드시 존재해야 하므로 폴백 폼으로 진행
                logger.warning(f"[WARN] form LLM failed. process={proc_def_id} activity={activity_id} err={e}")
                html = self._make_fallback_form_html()

            # 2) fields_json 추출
            try:
                fields_json = self._extract_fields_json_from_form_html(html)
            except Exception as e:
                logger.warning(f"[WARN] fields_json extract failed. fallback empty. err={e}")
                fields_json = []

            # 3) 저장
            ok = await self._save_form_def(
                form_def={
                    "id": form_def_id,
                    "html": html,
                    "proc_def_id": proc_def_id,
                    "activity_id": activity_id,
                    "fields_json": fields_json,
                },
                tenant_id=tenant_id,
            )
            if ok:
                forms_saved += 1

            await self._send_progress_event(
                event_queue,
                context_id,
                task_id,
                job_id,
                f"[FORM] 폼 저장 {'성공' if ok else '실패'}: {activity_name} (form_id={form_def_id})",
                "tool_usage_finished",
                95,
                {"proc_def_id": proc_def_id, "activity_id": activity_id, "form_def_id": form_def_id, "saved": ok},
            )

        return {"forms_saved": forms_saved, "activities": total}

    async def _update_proc_def_definition_only(self, *, proc_def_id: str, tenant_id: str, definition: Dict[str, Any]) -> bool:
        """proc_def.definition만 업데이트(폼 id 연결을 위해)."""
        if not self.supabase_client:
            return False
        try:
            # id는 tenant별 유니크라고 가정. (프론트도 id로 조회)
            self.supabase_client.table("proc_def").update(
                {
                    "definition": definition,
                    "tenant_id": tenant_id,
                    "isdeleted": False,
                }
            ).eq("id", proc_def_id).execute()
            return True
        except Exception as e:
            logger.warning(f"[WARN] proc_def.definition update failed: id={proc_def_id} err={e}")
            return False

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Query에서 PDF URL과 요청 정보를 파싱
        
        예시 입력:
        1. 순수 JSON: '{"pdf_url": "https://...", "description": "..."}'
        2. [InputData] JSON 형식:
           [InputData]
           {"path": "...", "fullPath": "http://...", "publicUrl": "http://...", "originalFileName": "..."}
        """
        result = {
            "pdf_url": "",
            "pdf_name": "",
            "description": "",
            "raw_query": query
        }
        
        # 1. 순수 JSON 형식 파싱 시도
        try:
            if query.strip().startswith('{'):
                data = json.loads(query)
                result["pdf_url"] = data.get("pdf_url", data.get("fileUrl", data.get("pdf_file_url", 
                                    data.get("fullPath", data.get("publicUrl", "")))))
                result["pdf_name"] = data.get("pdf_name", data.get("fileName", data.get("pdf_file_name",
                                    data.get("originalFileName", ""))))
                result["description"] = data.get("description", "")
                return result
        except json.JSONDecodeError:
            pass
        
        # 2. [InputData] 형식에서 JSON 추출
        if "[InputData]" in query:
            # [InputData] 다음의 JSON 객체 찾기
            input_data_match = re.search(r'\[InputData\]\s*(\{[^}]+\})', query, re.DOTALL)
            if input_data_match:
                try:
                    json_str = input_data_match.group(1)
                    data = json.loads(json_str)
                    # fullPath, publicUrl, path 순으로 URL 추출
                    result["pdf_url"] = data.get("fullPath", data.get("publicUrl", data.get("path", "")))
                    result["pdf_name"] = data.get("originalFileName", data.get("fileName", ""))
                    logger.info(f"[PARSE] Extracted from [InputData] JSON - URL: {result['pdf_url']}, Name: {result['pdf_name']}")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"[PARSE] Failed to parse [InputData] JSON: {e}")
            
            # key: value 형식 fallback
            url_match = re.search(r'pdf_file_url[:\s]+([^\s,]+)', query)
            if url_match:
                result["pdf_url"] = url_match.group(1).strip()
            
            name_match = re.search(r'pdf_file_name[:\s]+([^\s,]+)', query)
            if name_match:
                result["pdf_name"] = name_match.group(1).strip()
        
        # 3. URL 직접 추출 시도 (fallback)
        if not result["pdf_url"]:
            # .pdf로 끝나는 URL 또는 storage URL 찾기
            url_match = re.search(r'https?://[^\s<>"\'}\]]+(?:\.pdf|/storage/[^\s<>"\'}\]]+)', query, re.IGNORECASE)
            if url_match:
                result["pdf_url"] = url_match.group(0).rstrip('",')
                logger.info(f"[PARSE] Extracted URL via regex: {result['pdf_url']}")
        
        # 4. 파일명 추출 (URL에서)
        if result["pdf_url"] and not result["pdf_name"]:
            # URL에서 파일명 추출
            from urllib.parse import urlparse, unquote
            parsed = urlparse(result["pdf_url"])
            path_parts = parsed.path.split('/')
            if path_parts:
                result["pdf_name"] = unquote(path_parts[-1])
        
        return result

    async def _download_pdf(self, url: str, filename: str = None) -> str:
        """(deprecated) 파일 다운로드 후 임시 파일 경로 반환"""
        return await self._download_file(url, filename)

    async def _download_file(self, url: str, filename: str = None) -> str:
        """파일 다운로드 후 임시 파일 경로 반환 (PDF/Office/Image 등 모두 지원)"""
        client = await self._get_http_client()
        
        logger.info(f"[DOWNLOAD] Downloading file from: {url}")
        
        response = await client.get(url, follow_redirects=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
        
        # 임시 파일 생성
        suffix = ""
        if filename:
            # 파일명에서 확장자 추출
            p = Path(filename)
            suffix = p.suffix if p.suffix else ""
        if not suffix:
            # URL에서 확장자 추정
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                suffix = Path(parsed.path).suffix
            except Exception:
                suffix = ""
        if not suffix:
            suffix = ".bin"
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()
        
        logger.info(f"[DOWNLOAD] File saved to: {temp_file.name}")
        return temp_file.name

    def _normalize_text_key(self, s: str) -> str:
        return re.sub(r"\s+", "", (s or "").strip().lower())

    def _extract_teams_from_org_chart(self, chart: Dict[str, Any]) -> Dict[str, str]:
        """
        configuration(key=organization).value.chart 트리에서 팀(부서) 노드를 추출합니다.
        기대 구조(프론트 기준):
          { id, data: { isTeam: true, name }, children: [...] }
        """
        teams: Dict[str, str] = {}

        def walk(node: Any):
            if not node or not isinstance(node, dict):
                return
            node_id = str(node.get("id") or "")
            data = node.get("data") or {}
            if isinstance(data, dict) and data.get("isTeam"):
                name = str(data.get("name") or node_id or "").strip()
                if name and node_id:
                    teams[self._normalize_text_key(name)] = node_id
            children = node.get("children") or []
            if isinstance(children, list):
                for ch in children:
                    walk(ch)

        walk(chart)
        return teams

    async def _load_org_and_agents(self, tenant_id: str):
        """Supabase에서 조직도/에이전트 목록을 로드하여 캐시합니다."""
        if self._org_loaded:
            return
        self._org_loaded = True

        if not self.supabase_client:
            logger.warning("[WARN] Supabase client unavailable: org/agent mapping will be skipped.")
            return

        # 1) organization chart (teams)
        try:
            org = self.supabase_client.table("configuration").select("value").eq("key", "organization").eq("tenant_id", tenant_id).execute()
            if org.data and len(org.data) > 0:
                value = org.data[0].get("value")
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except Exception:
                        value = None
                if isinstance(value, dict):
                    chart = value.get("chart") or value
                    if isinstance(chart, dict):
                        self._org_teams_by_name = self._extract_teams_from_org_chart(chart)
        except Exception as e:
            logger.warning(f"[WARN] organization 로드 실패: {e}")

        # 2) agents (users table)
        try:
            agents = (
                self.supabase_client.table("users")
                .select("id, username, role, endpoint, agent_type, alias, is_agent")
                .eq("tenant_id", tenant_id)
                .eq("is_agent", True)
                .execute()
            )
            self._agents = agents.data or []
        except Exception as e:
            logger.warning(f"[WARN] agents(users) 로드 실패: {e}")

    def _pick_agent_for_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """역할명으로 users(is_agent=true) 중 가장 잘 맞는 agent를 선택."""
        key = self._normalize_text_key(role_name)
        if not key:
            return None

        # exact-ish match priority: username / role / alias
        for a in self._agents:
            if not isinstance(a, dict):
                continue
            if self._normalize_text_key(a.get("username")) == key:
                return a
            if self._normalize_text_key(a.get("role")) == key:
                return a
            if self._normalize_text_key(a.get("alias")) == key:
                return a

        # contains match
        for a in self._agents:
            if not isinstance(a, dict):
                continue
            cand = self._normalize_text_key(a.get("username")) or ""
            if cand and (cand in key or key in cand):
                return a
            cand = self._normalize_text_key(a.get("role")) or ""
            if cand and (cand in key or key in cand):
                return a
            cand = self._normalize_text_key(a.get("alias")) or ""
            if cand and (cand in key or key in cand):
                return a

        return None

    async def _send_progress_event(
        self, 
        event_queue: EventQueue, 
        context_id: str, 
        task_id: str,
        job_id: str,
        message: str,
        status: str,
        progress: int = 0,
        extra_data: Dict = None
    ):
        """진행 상황 이벤트 발송"""
        event_data = {
            "message": message,
            "status": status,
            "progress": progress,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if extra_data:
            event_data.update(extra_data)
        
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status={
                    "state": TaskState.working,
                    "message": new_agent_text_message(
                        json.dumps(event_data, ensure_ascii=False),
                        context_id, task_id
                    ),
                },
                final=False,
                contextId=context_id,
                taskId=task_id,
                metadata={
                    "crew_type": "pdf2bpmn",
                    "event_type": status,
                    "job_id": job_id,
                    "progress": progress
                }
            )
        )

    async def _send_bpmn_artifact(
        self,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        process_id: str,
        process_name: str,
        bpmn_xml: str,
        is_last: bool = False
    ):
        """BPMN XML 아티팩트 이벤트 발송"""
        artifact_data = {
            "type": "bpmn",
            "process_id": process_id,
            "process_name": process_name,
            "bpmn_xml": bpmn_xml,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                artifact=new_text_artifact(
                    name=f"BPMN: {process_name}",
                    description=f"Generated BPMN XML for process: {process_name}",
                    text=json.dumps(artifact_data, ensure_ascii=False),
                ),
                lastChunk=is_last,
                contextId=context_id,
                taskId=task_id,
            )
        )

    def _enrich_process_definition(
        self,
        proc_json: Dict[str, Any],
        *,
        process_name: str,
        process_definition_id: str,
    ) -> Dict[str, Any]:
        """
        proc_def.definition(JSON)이 "바로 실행 가능한 수준"에 가깝도록 최소 필드를 보정합니다.

        원칙:
        - 어떤 문서/형식이 와도 비어있지 않게(roles/activities/sequences 최소 1개) 보정
        - 추출/변환 결과를 최대한 존중하되, 필수 필드가 비면 안전한 기본값을 채움
        """
        # STRICT MODE:
        # - 문서에 없는 비즈니스 내용을 생성하지 않습니다.
        # - roles/tasks/events/sequences/data를 새로 "추가 생성"하지 않습니다.
        # - 단, 시스템 실행을 위한 기술적 필드(tool 등)는 비어있으면 기본값을 채울 수 있습니다.
        strict = os.getenv("STRICT_DEFINITION_MODE", "true").lower() == "true"

        result = proc_json or {}
        result["processDefinitionName"] = process_name or result.get("processDefinitionName") or "프로세스"
        result["processDefinitionId"] = process_definition_id or result.get("processDefinitionId") or ""

        # Ensure container keys exist
        for k in ("data", "roles", "events", "activities", "gateways", "sequences", "subProcesses", "participants"):
            if k not in result or result[k] is None:
                result[k] = []

        roles: List[Dict[str, Any]] = result.get("roles", []) if isinstance(result.get("roles"), list) else []
        activities: List[Dict[str, Any]] = result.get("activities", []) if isinstance(result.get("activities"), list) else []
        events: List[Dict[str, Any]] = result.get("events", []) if isinstance(result.get("events"), list) else []
        sequences: List[Dict[str, Any]] = result.get("sequences", []) if isinstance(result.get("sequences"), list) else []

        # STRICT: roles/participants 신규 생성 금지
        if strict:
            pass
        else:
            # Build role pool if missing (legacy behavior)
            if not roles:
                role_names = []
                for a in activities:
                    rn = (a.get("role") or "").strip()
                    if rn and rn not in role_names:
                        role_names.append(rn)
                if not role_names:
                    role_names = ["사용자"]
                roles = [{"name": rn, "endpoint": "", "resolutionRule": None, "default": ""} for rn in role_names]
                result["roles"] = roles

            # Ensure participants exist (Pool)
            participants: List[Dict[str, Any]] = result.get("participants", []) if isinstance(result.get("participants"), list) else []
            if not participants:
                participants = [{"id": f"Participant_{process_definition_id}", "name": result["processDefinitionName"], "processRef": result["processDefinitionId"]}]
                result["participants"] = participants

        primary_role = (roles[0].get("name") if roles else "") or ""

        # Build role lookup table
        role_by_name = {str(r.get("name", "")).strip(): r for r in roles if isinstance(r, dict) and r.get("name")}

        # Ensure each activity has required-ish fields
        for idx, a in enumerate(activities):
            if not isinstance(a, dict):
                continue
            a.setdefault("id", f"Activity_{idx+1}")
            a.setdefault("name", f"활동 {idx+1}")
            a.setdefault("type", a.get("type") or "userTask")
            # STRICT: role이 없으면 채우지 않음(문서 근거 없는 역할 생성 금지)
            if (not strict) and (not (a.get("role") or "").strip()) and primary_role:
                a["role"] = primary_role
            a.setdefault("description", "")
            a.setdefault("instruction", a.get("instruction") or a.get("description") or "")
            a.setdefault("duration", a.get("duration") or 5)

            # tool(form) - 없으면 안정적으로 생성
            if not (a.get("tool") or "").strip():
                safe_pid = re.sub(r"[^a-z0-9_]+", "_", (process_definition_id or "process").lower()).strip("_")
                safe_aid = re.sub(r"[^a-z0-9_]+", "_", str(a.get("id", f"activity_{idx+1}")).lower()).strip("_")
                a["tool"] = f"formHandler:{safe_pid}_{safe_aid}_form"

            # input/output data
            if not isinstance(a.get("inputData"), list):
                a["inputData"] = []
            if not isinstance(a.get("outputData"), list):
                a["outputData"] = []
            # STRICT: outputData 신규 생성 금지 (문서에 없는 데이터 변수 생성 금지)
            if not isinstance(a.get("checkpoints"), list):
                a["checkpoints"] = []

            # Agent execution fields (optional but makes process runnable)
            a.setdefault("agent", None)
            a.setdefault("agentMode", "none")
            a.setdefault("orchestration", None)
            a.setdefault("attachments", [])
            a.setdefault("customProperties", [])

        # STRICT: 이벤트 신규 생성 금지
        if not strict:
            def _has_event_type(type_name: str) -> bool:
                for e in events:
                    if isinstance(e, dict) and (e.get("type") == type_name):
                        return True
                return False

            if not _has_event_type("startEvent"):
                events.insert(0, {"id": "Event_Start", "name": "시작", "type": "startEvent", "role": primary_role, "process": process_definition_id})
            if not _has_event_type("endEvent"):
                events.append({"id": "Event_End", "name": "종료", "type": "endEvent", "role": primary_role, "process": process_definition_id})
            result["events"] = events

        # STRICT: 시퀀스/데이터 신규 생성 금지.
        # 다만 XML→JSON 변환 결과가 condition을 name에 넣었을 경우, condition 복원은 "내용 생성"이 아니라 필드 정규화로 간주.
        for s in sequences:
            if not isinstance(s, dict):
                continue
            if (not s.get("condition")) and s.get("name"):
                s["condition"] = s.get("name")
        result["sequences"] = sequences

        return result

    def _convert_xml_to_json(self, bpmn_xml: str) -> Dict[str, Any]:
        """
        BPMN XML을 ProcessGPT JSON 형식으로 변환
        ProcessDefinitionModule.vue의 convertXMLToJSON과 유사한 로직
        """
        try:
            # Prefer robust converter (ported from old_pdf2bpmn) if available.
            # 로컬/컨테이너 어디서 실행되든 `src/`를 sys.path에 추가해 import 가능하게 합니다.
            try:
                repo_root = Path(__file__).resolve().parent
                src_dir = str(repo_root / "src")
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from pdf2bpmn.bpmn_to_json import BPMNToJSONConverter  # type: ignore

                converter = BPMNToJSONConverter()
                # 아래 2개 값은 호출자가 바깥에서 세팅하므로 여기서는 더미로 채움
                return converter.convert(bpmn_xml, process_definition_id="", process_name="")
            except Exception as e:
                logger.warning(f"[WARN] 고급 BPMN→JSON 변환기 로드 실패. 단순 변환으로 fallback 합니다. err={e}")

            root = ET.fromstring(bpmn_xml)
            
            # 네임스페이스 처리
            namespaces = {
                'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
                'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'
            }
            
            result = {
                "processDefinitionId": "",
                "processDefinitionName": "",
                "version": "1.0",
                "shortDescription": "",
                "description": "",
                "data": [],
                "roles": [],
                "events": [],
                "activities": [],
                "gateways": [],
                "sequences": [],
                "subProcesses": []
            }
            
            # Process 정보 추출
            process = root.find('.//bpmn:process', namespaces)
            if process is None:
                # 네임스페이스 없는 경우
                process = root.find('.//process')
            
            if process is not None:
                result["processDefinitionId"] = process.get('id', '')
                result["processDefinitionName"] = process.get('name', '')
            
            # Participants에서 이름 추출 시도
            collaboration = root.find('.//bpmn:collaboration', namespaces)
            if collaboration is None:
                collaboration = root.find('.//collaboration')
            
            if collaboration is not None:
                participant = collaboration.find('.//bpmn:participant', namespaces)
                if participant is None:
                    participant = collaboration.find('.//participant')
                if participant is not None:
                    result["processDefinitionName"] = participant.get('name', result["processDefinitionName"])
            
            # Lanes (Roles) 추출
            lanes = root.findall('.//bpmn:lane', namespaces)
            if not lanes:
                lanes = root.findall('.//lane')
            
            for lane in lanes:
                role = {
                    "name": lane.get('name', ''),
                    "endpoint": "",
                    "resolutionRule": "",
                    "default": ""
                }
                result["roles"].append(role)
            
            # Tasks (Activities) 추출
            task_types = ['userTask', 'serviceTask', 'task', 'manualTask', 'scriptTask']
            for task_type in task_types:
                tasks = root.findall(f'.//bpmn:{task_type}', namespaces)
                if not tasks:
                    tasks = root.findall(f'.//{task_type}')
                
                for task in tasks:
                    activity = {
                        "id": task.get('id', ''),
                        "name": task.get('name', ''),
                        "type": task_type,
                        "description": "",
                        "instruction": "",
                        "role": "",
                        "tool": "formHandler:defaultform",
                        "duration": 5
                    }
                    result["activities"].append(activity)
            
            # Events 추출
            for event_type in ['startEvent', 'endEvent', 'intermediateThrowEvent', 'intermediateCatchEvent']:
                events = root.findall(f'.//bpmn:{event_type}', namespaces)
                if not events:
                    events = root.findall(f'.//{event_type}')
                
                for event in events:
                    evt = {
                        "id": event.get('id', ''),
                        "name": event.get('name', ''),
                        "type": event_type,
                        "role": "",
                        "process": result["processDefinitionId"]
                    }
                    result["events"].append(evt)
            
            # Gateways 추출
            for gateway_type in ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway']:
                gateways = root.findall(f'.//bpmn:{gateway_type}', namespaces)
                if not gateways:
                    gateways = root.findall(f'.//{gateway_type}')
                
                for gateway in gateways:
                    gw = {
                        "id": gateway.get('id', ''),
                        "name": gateway.get('name', ''),
                        "type": gateway_type,
                        "condition": ""
                    }
                    result["gateways"].append(gw)
            
            # Sequence Flows 추출
            sequences = root.findall('.//bpmn:sequenceFlow', namespaces)
            if not sequences:
                sequences = root.findall('.//sequenceFlow')
            
            for seq in sequences:
                sequence = {
                    "id": seq.get('id', ''),
                    "name": seq.get('name', ''),
                    "source": seq.get('sourceRef', ''),
                    "target": seq.get('targetRef', ''),
                    "condition": ""
                }
                result["sequences"].append(sequence)
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] XML to JSON conversion failed: {e}")
            return {
                "processDefinitionId": f"process_{uuid.uuid4().hex[:8]}",
                "processDefinitionName": "Converted Process",
                "data": [],
                "roles": [],
                "events": [],
                "activities": [],
                "gateways": [],
                "sequences": []
            }

    async def _save_proc_def(self, proc_def: Dict, tenant_id: str) -> bool:
        """프로세스 정의를 proc_def 테이블에 저장"""
        if not self.supabase_client:
            logger.error("[ERROR] Supabase client is None! Cannot save proc_def")
            return False
        
        try:
            logger.info(f"[DB-PROC_DEF] ========== START ==========")
            logger.info(f"[DB-PROC_DEF] id={proc_def['id']}, tenant_id={tenant_id}")
            logger.info(f"[DB-PROC_DEF] name={proc_def.get('name')}, bpmn_length={len(proc_def.get('bpmn', ''))}")
            logger.info(f"[DB-PROC_DEF] definition keys: {list(proc_def.get('definition', {}).keys()) if proc_def.get('definition') else 'None'}")
            
            # 기존 proc_def 확인
            logger.info(f"[DB-PROC_DEF] Checking existing...")
            existing = self.supabase_client.table('proc_def').select('id, uuid').eq('id', proc_def['id']).execute()
            logger.info(f"[DB-PROC_DEF] Existing result: {existing.data}")
            
            if existing.data and len(existing.data) > 0:
                existing_uuid = existing.data[0].get('uuid')
                logger.info(f"[DB-PROC_DEF] Updating existing uuid={existing_uuid}")
                result = self.supabase_client.table('proc_def').update({
                    'name': proc_def['name'],
                    'definition': proc_def['definition'],
                    'bpmn': proc_def['bpmn'],
                    'type': proc_def.get('type', 'bpmn'),
                    'isdeleted': False,
                    'tenant_id': tenant_id
                }).eq('uuid', existing_uuid).execute()
                logger.info(f"[DB-PROC_DEF] Update result: {result.data}")
            else:
                insert_data = {
                    'id': proc_def['id'],
                    'name': proc_def['name'],
                    'definition': proc_def['definition'],
                    'bpmn': proc_def['bpmn'],
                    'tenant_id': tenant_id,
                    'type': proc_def.get('type', 'bpmn'),
                    'isdeleted': False
                }
                
                logger.info(f"[DB-PROC_DEF] Inserting new record...")
                logger.info(f"[DB-PROC_DEF] Insert data keys: {list(insert_data.keys())}")
                result = self.supabase_client.table('proc_def').insert(insert_data).execute()
                logger.info(f"[DB-PROC_DEF] Insert result: {result.data}")
            
            logger.info(f"[DB-PROC_DEF] ========== SUCCESS ==========")
            return True
            
        except Exception as e:
            logger.error(f"[DB-PROC_DEF] ========== ERROR ==========")
            logger.error(f"[DB-PROC_DEF] Exception type: {type(e).__name__}")
            logger.error(f"[DB-PROC_DEF] Exception message: {e}")
            import traceback
            logger.error(f"[DB-PROC_DEF] Traceback:\n{traceback.format_exc()}")
            return False

    async def _update_proc_map(self, new_process: Dict, tenant_id: str) -> bool:
        """
        configuration 테이블의 proc_map 업데이트
        미분류 카테고리에 새 프로세스 추가
        """
        if not self.supabase_client:
            logger.warning("[WARN] Supabase client not available, skipping proc_map update")
            return False
        
        try:
            # 기존 proc_map 조회
            result = self.supabase_client.table('configuration').select('value').eq('key', 'proc_map').eq('tenant_id', tenant_id).execute()
            
            if result.data and len(result.data) > 0:
                proc_map = result.data[0].get('value', {})
            else:
                # proc_map이 없으면 새로 생성
                proc_map = {"mega_proc_list": []}
            
            if not isinstance(proc_map, dict):
                proc_map = {"mega_proc_list": []}
            
            mega_proc_list = proc_map.get('mega_proc_list', [])
            
            # 미분류 메가 프로세스 찾기
            unclassified_mega = None
            for mega in mega_proc_list:
                if mega.get('id') == 'unclassified' or mega.get('name') == '미분류':
                    unclassified_mega = mega
                    break
            
            if not unclassified_mega:
                # 미분류 메가 프로세스 생성
                unclassified_mega = {
                    "id": "unclassified",
                    "name": "미분류",
                    "major_proc_list": []
                }
                mega_proc_list.append(unclassified_mega)
            
            # 미분류 메이저 프로세스 찾기
            major_proc_list = unclassified_mega.get('major_proc_list', [])
            unclassified_major = None
            for major in major_proc_list:
                if major.get('id') == 'unclassified_major' or major.get('name') == '미분류':
                    unclassified_major = major
                    break
            
            if not unclassified_major:
                # 미분류 메이저 프로세스 생성
                unclassified_major = {
                    "id": "unclassified_major",
                    "name": "미분류",
                    "sub_proc_list": []
                }
                major_proc_list.append(unclassified_major)
                unclassified_mega['major_proc_list'] = major_proc_list
            
            # 서브 프로세스 목록에 추가 (중복 체크)
            sub_proc_list = unclassified_major.get('sub_proc_list', [])
            exists = any(p.get('id') == new_process['id'] for p in sub_proc_list)
            
            if not exists:
                sub_proc_list.append({
                    "id": new_process['id'],
                    "name": new_process['name'],
                    "path": new_process['id'],
                    "new": True
                })
                unclassified_major['sub_proc_list'] = sub_proc_list
            
            proc_map['mega_proc_list'] = mega_proc_list
            
            # configuration 테이블 업데이트
            if result.data and len(result.data) > 0:
                self.supabase_client.table('configuration').update({
                    'value': proc_map
                }).eq('key', 'proc_map').eq('tenant_id', tenant_id).execute()
            else:
                self.supabase_client.table('configuration').insert({
                    'key': 'proc_map',
                    'value': proc_map,
                    'tenant_id': tenant_id
                }).execute()
            
            logger.info(f"[DB] Updated proc_map with process: {new_process['id']}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update proc_map: {e}")
            return False

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        메인 실행 로직 - ProcessGPT SDK 인터페이스 구현
        
        Args:
            context: 요청 컨텍스트 (사용자 입력, 컨텍스트 데이터 포함)
            event_queue: 이벤트 큐 (진행 상황 및 결과 전송용)
        """
        # 1. 작업 정보 가져오기
        logger.info(f"[DEBUG] context: {context}")
        
        context_data = context.get_context_data()
        row = context_data.get("row", {})
        logger.info(f"[DEBUG] row: {row}")
        logger.info(f"[DEBUG] context_data keys: {context_data.keys()}")
        
        task_id = row.get("id")
        # context_id가 None이면 task_id를 사용 (adhoc task의 경우)
        context_id = row.get("root_proc_inst_id") or row.get("proc_inst_id") or task_id
        tenant_id = row.get("tenant_id", "uengine")
        
        # Query 가져오기 - 여러 소스에서 시도
        user_input = context.get_user_input()
        logger.info(f"[DEBUG] context.get_user_input(): '{user_input[:200] if user_input else 'None'}...'")
        
        # context_data에서 query 확인
        if not user_input and context_data.get('query'):
            user_input = context_data.get('query')
            logger.info(f"[INFO] Got user_input from context_data.query: '{user_input[:100]}...'")
        
        # row에서 query 확인
        if not user_input and row.get('query'):
            user_input = row.get('query')
            logger.info(f"[INFO] Got user_input from row.query: '{user_input[:100]}...'")
        
        # description fallback
        if not user_input and row.get('description'):
            user_input = row.get('description')
            logger.info(f"[INFO] Got user_input from description: '{user_input[:100]}...'")
        
        # Job ID 생성
        job_id = f"pdf2bpmn-{task_id}"
        
        logger.info(f"[START] PDF2BPMN task: {user_input[:100] if user_input else 'N/A'}... (job_id: {job_id})")
        
        temp_pdf_path = None
        
        try:
            # 2. 작업 시작 이벤트
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[START] PDF2BPMN 변환 작업을 시작합니다...",
                "task_started", 0
            )
            
            # 3. Query 파싱 (PDF 정보 추출)
            parsed = self._parse_query(user_input or "")
            pdf_name = parsed.get("pdf_name", "document.pdf")
            logger.info(f"[INFO] PDF Name: {pdf_name}")
            
            # # 4~7. PDF 다운로드/업로드/처리 (주석처리 - 프론트에서 이미 처리됨)
            # pdf_url = parsed.get("pdf_url", "")
            # if not pdf_url:
            #     raise Exception("PDF URL이 제공되지 않았습니다.")
            # await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #     f"[DOWNLOAD] PDF 파일 다운로드 중: {pdf_name}", "tool_usage_started", 5)
            # temp_pdf_path = await self._download_pdf(pdf_url, pdf_name)
            # await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #     "[UPLOAD] PDF 파일을 분석 서버에 업로드 중...", "tool_usage_started", 10)
            # client = await self._get_http_client()
            # with open(temp_pdf_path, 'rb') as f:
            #     files = {'file': (pdf_name, f, 'application/pdf')}
            #     upload_response = await client.post(f"{self.pdf2bpmn_url}/api/upload", files=files)
            # if upload_response.status_code != 200:
            #     raise Exception(f"PDF 업로드 실패: {upload_response.status_code}")
            # upload_result = upload_response.json()
            # processing_job_id = upload_result.get("job_id")
            # await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #     "[PROCESSING] PDF 분석 및 BPMN 변환을 시작합니다...", "tool_usage_started", 15)
            # process_response = await client.post(f"{self.pdf2bpmn_url}/api/process/{processing_job_id}")
            # if process_response.status_code != 200:
            #     raise Exception(f"처리 시작 실패: {process_response.status_code}")
            # max_retries = 600
            # retry_count = 0
            # last_progress = 15
            # while retry_count < max_retries:
            #     if self.is_cancelled:
            #         raise Exception("작업이 취소되었습니다.")
            #     status_response = await client.get(f"{self.pdf2bpmn_url}/api/jobs/{processing_job_id}")
            #     if status_response.status_code != 200:
            #         raise Exception(f"상태 조회 실패: {status_response.status_code}")
            #     job_status = status_response.json()
            #     current_status = job_status.get("status", "")
            #     current_progress = job_status.get("progress", 0)
            #     detail_message = job_status.get("detail_message", "")
            #     chunk_info = job_status.get("chunk_info")
            #     if retry_count % 5 == 0:
            #         logger.info(f"[POLL] status={current_status}, progress={current_progress}")
            #     if current_status == "completed":
            #         logger.info("[INFO] Processing completed")
            #         break
            #     elif current_status == "error":
            #         error_msg = job_status.get("error", "알 수 없는 오류")
            #         raise Exception(f"처리 중 오류 발생: {error_msg}")
            #     mapped_progress = 15 + int(current_progress * 0.7)
            #     if current_progress != last_progress:
            #         extra_data = {}
            #         if chunk_info:
            #             extra_data["chunk_info"] = chunk_info
            #         await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #             f"[PROCESSING] {detail_message or f'진행 중... ({current_progress}%)'}", 
            #             "tool_usage_started", mapped_progress, extra_data)
            #         last_progress = current_progress
            #     await asyncio.sleep(1)
            #     retry_count += 1
            # if retry_count >= max_retries:
            #     raise Exception("처리 시간 초과")
            
            client = await self._get_http_client()
            
            # =================================================================
            # 4. PDF URL로 처리 시작
            # =================================================================
            pdf_url = parsed.get("pdf_url", "")
            if not pdf_url:
                raise Exception("PDF URL이 제공되지 않았습니다. query에 pdf_url을 포함해주세요.")
            
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                f"[UPLOAD] PDF 파일을 분석 서버에 업로드 중: {pdf_name}",
                "tool_usage_started", 10
            )
            
            # 파일 다운로드 (pdf/xlsx/pptx/image 등 모두 허용)
            temp_pdf_path = await self._download_file(pdf_url, pdf_name)
            
            # PDF2BPMN API에 업로드
            with open(temp_pdf_path, 'rb') as f:
                # Content-Type은 일반적으로 octet-stream으로 보내고,
                # 서버가 filename 확장자를 보고 PDF 변환 여부를 결정합니다.
                files = {'file': (pdf_name, f, 'application/octet-stream')}
                upload_response = await client.post(f"{self.pdf2bpmn_url}/api/upload", files=files)
            
            if upload_response.status_code != 200:
                raise Exception(f"PDF 업로드 실패: {upload_response.status_code} - {upload_response.text}")
            
            upload_result = upload_response.json()
            processing_job_id = upload_result.get("job_id")
            logger.info(f"[INFO] PDF 업로드 완료, job_id: {processing_job_id}")
            
            # =================================================================
            # 5. PDF 처리 시작 및 진행 상황 폴링
            # =================================================================
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[PROCESSING] PDF 분석 및 BPMN 변환을 시작합니다...",
                "tool_usage_started", 15
            )
            
            process_response = await client.post(f"{self.pdf2bpmn_url}/api/process/{processing_job_id}")
            if process_response.status_code != 200:
                raise Exception(f"처리 시작 실패: {process_response.status_code}")
            
            # 진행 상황 폴링 (self.timeout 사용 - config에서 전달받은 값, 기본 1시간)
            max_retries = self.timeout  # 1초 간격으로 폴링
            retry_count = 0
            last_progress = 15
            logger.info(f"[INFO] PDF 처리 폴링 시작 (timeout: {self.timeout}초)")
            
            while retry_count < max_retries:
                if self.is_cancelled:
                    raise Exception("작업이 취소되었습니다.")
                
                status_response = await client.get(f"{self.pdf2bpmn_url}/api/jobs/{processing_job_id}")
                if status_response.status_code != 200:
                    raise Exception(f"상태 조회 실패: {status_response.status_code}")
                
                job_status = status_response.json()
                current_status = job_status.get("status", "")
                current_progress = job_status.get("progress", 0)
                detail_message = job_status.get("detail_message", "")
                
                if retry_count % 10 == 0:  # 10초마다 로그
                    logger.info(f"[POLL] status={current_status}, progress={current_progress}")
                
                if current_status == "completed":
                    logger.info("[INFO] Processing completed")
                    break
                elif current_status == "error":
                    error_msg = job_status.get("error", "알 수 없는 오류")
                    raise Exception(f"처리 중 오류 발생: {error_msg}")
                
                # 진행률 이벤트 (변경 시에만)
                mapped_progress = 15 + int(current_progress * 0.7)  # 15% ~ 85%
                if current_progress != last_progress:
                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        f"[PROCESSING] {detail_message or f'진행 중... ({current_progress}%)'}",
                        "tool_usage_started", mapped_progress
                    )
                    last_progress = current_progress
                
                await asyncio.sleep(1)
                retry_count += 1
            
            if retry_count >= max_retries:
                raise Exception("처리 시간 초과")
            
            # =================================================================
            # 6. 결과 가져오기 - processing_job_id 기준으로 "이번 작업" 결과만 조회
            # =================================================================
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[GENERATING] 이번 작업의 BPMN XML들을 가져옵니다...",
                "tool_usage_started", 88
            )
            # 6-1) 작업(job) 상태에서 이번 처리 결과(process_id 목록)를 가져온다.
            job_process_ids: List[str] = []
            process_names_by_id: Dict[str, str] = {}
            last_result_info: Dict[str, Any] = {}
            try:
                # API 쪽에서 status=completed와 result 세팅 타이밍 레이스가 있을 수 있어
                # process_ids가 채워질 때까지 짧게 재시도합니다.
                for attempt in range(1, 11):  # 최대 약 5초(0.5s * 10) 대기
                    job_status_response = await client.get(f"{self.pdf2bpmn_url}/api/jobs/{processing_job_id}")
                    if job_status_response.status_code != 200:
                        logger.warning(
                            f"[WARN] job status 조회 실패: {job_status_response.status_code} - {job_status_response.text}"
                        )
                        break

                    job_status = job_status_response.json()
                    result_info = (job_status.get("result") or {})
                    if isinstance(result_info, dict):
                        last_result_info = result_info

                    # 우선순위: process_ids (job 단위 스코핑) -> bpmn_files keys (구버전/옵션)
                    result_process_ids = result_info.get("process_ids")
                    if isinstance(result_process_ids, list) and result_process_ids:
                        job_process_ids = [str(pid) for pid in result_process_ids if pid]
                    else:
                        job_bpmn_files = (result_info.get("bpmn_files") or {})
                        if isinstance(job_bpmn_files, dict) and job_bpmn_files:
                            job_process_ids = [str(pid) for pid in job_bpmn_files.keys() if pid]

                    pn = result_info.get("process_names")
                    if isinstance(pn, dict) and not process_names_by_id:
                        process_names_by_id = {str(k): str(v) for k, v in pn.items() if k and v}

                    if job_process_ids:
                        if attempt > 1:
                            logger.info(f"[INFO] job result process_ids 확인됨 (attempt={attempt})")
                        break

                    # 아직 result가 비어있을 수 있음(레이스) → 잠깐 대기 후 재시도
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"[WARN] job status 조회 중 예외: {e}")

            # 6-2) job_id 기반 process_id들로 BPMN content를 개별 조회한다.
            bpmn_files: Dict[str, Dict[str, Any]] = {}
            if not job_process_ids:
                # 이미지 슬라이드/PPT 등 "프로세스 추출이 0개"인 케이스는 정상 성공(0개)로 처리합니다.
                # - result.processes(=추출된 프로세스 수)가 0이면 생성할 BPMN이 없음을 사용자에게 안내
                # - 그 외에는 시스템 오류로 보고 실패 처리
                extracted_count = None
                if isinstance(last_result_info, dict):
                    extracted_count = last_result_info.get("processes")
                    if extracted_count is None:
                        extracted_count = last_result_info.get("process_count")

                if extracted_count == 0:
                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        "[NOTICE] 문서에서 추출된 프로세스가 없어 생성할 BPMN이 없습니다. (이미지/슬라이드 위주 문서일 수 있습니다.)",
                        "tool_usage_finished", 100,
                        {"process_count": 0, "reason": "no_process_extracted"}
                    )
                    # bpmn_files는 빈 상태로 유지하고, 이후 로직에서 0개 결과로 정상 완료 메시지를 발송합니다.
                else:
                    raise Exception(
                        "이번 처리(job_id)에서 생성된 process_id 목록을 얻지 못했습니다. "
                        "PDF2BPMN API가 /api/jobs/{job_id}.result.process_ids(권장) 또는 result.bpmn_files를 제공해야 합니다."
                    )

            if job_process_ids:
                logger.info(f"[INFO] 이번 작업(processing_job_id={processing_job_id}) 프로세스 {len(job_process_ids)}개 감지")
                for proc_id in job_process_ids:
                    try:
                        content_response = await client.get(
                            f"{self.pdf2bpmn_url}/api/files/bpmn/content?process_id={proc_id}"
                        )
                        if content_response.status_code != 200:
                            logger.warning(
                                f"[WARN] BPMN content 조회 실패: proc_id={proc_id}, status={content_response.status_code}"
                            )
                            continue
                        content_result = content_response.json()
                        bpmn_files[proc_id] = {
                            "content": content_result.get("content", ""),
                            "process_name": content_result.get("process_name") or process_names_by_id.get(proc_id) or "",
                            "filename": None,
                        }
                    except Exception as e:
                        logger.warning(f"[WARN] BPMN content 조회 중 예외: proc_id={proc_id}, err={e}")

            bpmn_count = len(bpmn_files)
            logger.info(f"[INFO] 이번 작업 기준 BPMN: {bpmn_count}개")
            
            # 9. 각 BPMN에 대해 이벤트 발송 및 DB 저장
            saved_processes = []
            all_bpmn_xmls = {}  # proc_def_id -> bpmn_xml 매핑
            total_bpmn = len(bpmn_files)
            
            logger.info(f"[DEBUG] bpmn_files keys: {list(bpmn_files.keys())}")
            
            for idx, (proc_id, bpmn_data) in enumerate(bpmn_files.items()):
                bpmn_xml = bpmn_data.get("content", "")
                process_name = bpmn_data.get("process_name", f"Process {idx + 1}")
                
                logger.info(f"[DEBUG] Processing BPMN {idx+1}/{total_bpmn}: {process_name}")
                logger.info(f"[DEBUG] BPMN XML length: {len(bpmn_xml)} chars")
                
                if not bpmn_xml:
                    logger.warning(f"[WARN] Empty BPMN XML for process: {process_name}")
                    continue
                
                # XML을 JSON으로 변환
                proc_json = self._convert_xml_to_json(bpmn_xml)
                proc_json["processDefinitionName"] = process_name
                
                # proc_def_id 생성 (안전한 형식)
                safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', process_name.lower())[:50]
                proc_def_id = f"{safe_id}_{proc_id[:8]}"
                
                proc_json["processDefinitionId"] = proc_def_id

                # 조직도/에이전트 정보를 로드해 역할→에이전트 매핑까지 반영
                await self._load_org_and_agents(tenant_id)

                # 실행 가능한 수준으로 최소 보정(roles/activities/tool/outputData/sequences 등)
                proc_json = self._enrich_process_definition(
                    proc_json,
                    process_name=process_name,
                    process_definition_id=proc_def_id,
                )

                # roles/activities에 endpoint/agent/orchestration을 채움 (가능한 경우)
                try:
                    roles = proc_json.get("roles", [])
                    activities = proc_json.get("activities", [])
                    if isinstance(roles, list):
                        for r in roles:
                            if not isinstance(r, dict):
                                continue
                            rn = str(r.get("name") or "").strip()
                            if not rn:
                                continue

                            # 1) agent 우선 매핑
                            agent = self._pick_agent_for_role(rn)
                            if agent and agent.get("id"):
                                r["endpoint"] = agent.get("id")
                                r["origin"] = "used"
                                continue

                            # 2) team 매핑 (조직도)
                            team_id = self._org_teams_by_name.get(self._normalize_text_key(rn))
                            if team_id:
                                r["endpoint"] = team_id
                                r["origin"] = "used"
                                continue

                            # 3) 생성된 역할
                            r.setdefault("endpoint", "")
                            r["origin"] = r.get("origin") or "created"

                    # activity agent mapping
                    if isinstance(activities, list):
                        default_agent_mode = os.getenv("DEFAULT_AGENT_MODE", "draft")
                        for a in activities:
                            if not isinstance(a, dict):
                                continue
                            role_name = str(a.get("role") or "").strip()
                            if not role_name:
                                continue

                            agent = self._pick_agent_for_role(role_name)
                            if not agent or not agent.get("id"):
                                # human/user activity
                                a["agent"] = None
                                a["agentMode"] = "none"
                                a["orchestration"] = None
                                continue

                            a["agent"] = agent.get("id")
                            a["agentMode"] = default_agent_mode

                            agent_type = (agent.get("agent_type") or "agent").lower()
                            if agent_type == "agent":
                                a["orchestration"] = "crewai-action"
                            elif agent_type == "a2a":
                                a["orchestration"] = "a2a"
                            elif agent_type == "pgagent":
                                a["orchestration"] = agent.get("alias") or "pgagent"
                            else:
                                a["orchestration"] = agent.get("alias") or agent_type
                except Exception as e:
                    logger.warning(f"[WARN] role/agent 매핑 보정 실패: {e}")
                
                # BPMN XML 저장
                all_bpmn_xmls[proc_def_id] = bpmn_xml
                
                # DB에 저장
                proc_def_data = {
                    "id": proc_def_id,
                    "name": process_name,
                    "definition": proc_json,
                    "bpmn": bpmn_xml,
                    "uuid": str(uuid.uuid4()),
                    "type": "bpmn",
                    "owner": None,
                    "prod_version": None
                }
                
                # proc_def 테이블 저장
                save_result = await self._save_proc_def(proc_def_data, tenant_id)
                logger.info(f"[DEBUG] proc_def save result: {save_result}")
                
                # proc_map 업데이트
                await self._update_proc_map({"id": proc_def_id, "name": process_name}, tenant_id)

                # -----------------------------------------------------------------
                # B안: proc_def 먼저 저장 → 폼 생성/저장(프론트 없이도 워커가 수행)
                # - 실패해도 폴백 폼을 만들어 form_def에 저장 시도
                # -----------------------------------------------------------------
                if save_result:
                    try:
                        await self._send_progress_event(
                            event_queue, context_id, task_id, job_id,
                            f"[FORM] 프로세스 폼 생성/저장을 시작합니다: {process_name}",
                            "tool_usage_started", 91,
                            {"proc_def_id": proc_def_id, "process_name": process_name},
                        )
                        forms_result = await self._ensure_forms_for_process(
                            proc_def_id=proc_def_id,
                            process_name=process_name,
                            proc_json=proc_json,
                            tenant_id=tenant_id,
                            event_queue=event_queue,
                            context_id=context_id,
                            task_id=task_id,
                            job_id=job_id,
                        )
                        # 폼 id를 activity.tool에 반영했으므로, proc_def.definition도 동기화 업데이트
                        await self._update_proc_def_definition_only(
                            proc_def_id=proc_def_id,
                            tenant_id=tenant_id,
                            definition=proc_json,
                        )
                        await self._send_progress_event(
                            event_queue, context_id, task_id, job_id,
                            f"[FORM] 프로세스 폼 처리 완료: {process_name} (saved={forms_result.get('forms_saved')}/{forms_result.get('activities')})",
                            "tool_usage_finished", 96,
                            {"proc_def_id": proc_def_id, "forms_result": forms_result},
                        )
                    except Exception as e:
                        logger.warning(f"[WARN] form generation/save stage failed unexpectedly: {e}")
                
                # saved_processes에 bpmn_xml 포함
                saved_processes.append({
                    "id": proc_def_id,
                    "name": process_name,
                    "bpmn_xml": bpmn_xml  # XML 내용 포함
                })
                
                # 진행 이벤트 (XML 포함)
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    f"[SAVED] 프로세스 저장 완료: {process_name}",
                    "tool_usage_finished", 90 + int(10 * (idx + 1) / total_bpmn),
                    {
                        "process_id": proc_def_id, 
                        "process_name": process_name,
                        "bpmn_xml": bpmn_xml  # 이벤트에도 XML 포함
                    }
                )
            
            # 10. 최종 결과 구성 (saved_processes에는 이미 bpmn_xml 포함됨)
            actual_count = len(saved_processes)
            logger.info(f"[DEBUG] Actual saved process count: {actual_count}")
            
            completed_message = (
                "[COMPLETED] PDF2BPMN 변환 완료: 문서에서 프로세스를 추출하지 못해 생성할 BPMN이 없습니다."
                if actual_count == 0
                else f"[COMPLETED] PDF2BPMN 변환 완료: {actual_count}개의 프로세스가 생성되었습니다."
            )

            final_result = {
                "message": completed_message,
                "status": "completed",
                "job_id": job_id,
                "pdf_name": pdf_name,
                "process_count": actual_count,
                "saved_processes": saved_processes,  # bpmn_xml 포함
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # 11. 최종 결과 아티팩트 이벤트 (browser_use와 동일한 패턴)
            # 이 이벤트가 프론트엔드에서 최종 결과로 사용됨
            # saved_processes에서 요약 정보만 추출 (draft 크기 제한 고려)
            saved_processes_summary = [
                {"id": p["id"], "name": p["name"]} for p in saved_processes
            ]
            
            final_artifact_data = {
                "type": "pdf2bpmn_result",
                "pdf_name": pdf_name,
                "process_count": actual_count,
                "saved_processes": saved_processes_summary,  # 요약만
                "bpmn_xmls": all_bpmn_xmls,  # 모든 XML 내용
                "success": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "task_type": "pdf2bpmn"
            }
            
            event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="PDF2BPMN Result",
                        description=f"PDF2BPMN 변환 결과: {actual_count}개 프로세스 생성",
                        text=json.dumps(final_artifact_data, ensure_ascii=False),
                    ),
                    lastChunk=True,  # 최종 결과 표시
                    contextId=context_id,
                    taskId=task_id,
                )
            )
            
            # 12. 완료 상태 이벤트
            event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status={
                        "state": TaskState.working,
                        "message": new_agent_text_message(
                            json.dumps(final_result, ensure_ascii=False),
                            context_id, task_id
                        ),
                    },
                    final=True,
                    contextId=context_id,
                    taskId=task_id,
                    metadata={
                        "crew_type": "pdf2bpmn",
                        "event_type": "task_completed",
                        "job_id": job_id,
                        "process_count": actual_count
                    }
                )
            )
            
            logger.info(f"[DONE] Task completed: {job_id} ({bpmn_count} processes)")
            
        except httpx.ConnectError as e:
            logger.error(f"[ERROR] Cannot connect to PDF2BPMN server: {e}")
            error_msg = f"PDF2BPMN 서버에 연결할 수 없습니다: {self.pdf2bpmn_url}. 서버가 실행 중인지 확인하세요."
            await self._send_error_event(event_queue, context_id, task_id, job_id, error_msg, "connection_error")
            
        except Exception as e:
            logger.error(f"[ERROR] Task execution error: {e}")
            logger.error(traceback.format_exc())
            await self._send_error_event(event_queue, context_id, task_id, job_id, str(e), type(e).__name__)
        
        finally:
            # 임시 파일 정리
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                    logger.info(f"[CLEANUP] Removed temp file: {temp_pdf_path}")
                except Exception as e:
                    logger.warning(f"[WARN] Failed to remove temp file: {e}")
            
            # HTTP 클라이언트 정리
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None

    async def _send_error_event(
        self, 
        event_queue: EventQueue, 
        context_id: str, 
        task_id: str, 
        job_id: str, 
        error_msg: str, 
        error_type: str
    ):
        """에러 이벤트 발송"""
        error_data = {
            "message": f"[ERROR] PDF2BPMN 작업 실패: {error_msg}",
            "error": error_msg,
            "error_type": error_type,
            "status": "failed",
            "job_id": job_id,
            "pdf2bpmn_url": self.pdf2bpmn_url
        }
        
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status={
                    "state": TaskState.working,
                    "message": new_agent_text_message(
                        json.dumps(error_data, ensure_ascii=False),
                        context_id, task_id
                    ),
                },
                final=True,
                contextId=context_id,
                taskId=task_id,
                metadata={
                    "crew_type": "pdf2bpmn",
                    "event_type": "error",
                    "job_id": job_id
                }
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 처리"""
        self.is_cancelled = True
        
        row = context.get_context_data().get("row", {})
        context_id = row.get("root_proc_inst_id") or row.get("proc_inst_id")
        task_id = row.get("id")
        
        cancel_data = {
            "message": "[CANCELLED] PDF2BPMN 작업이 취소되었습니다.",
            "status": "cancelled"
        }
        
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status={
                    "state": TaskState.working,
                    "message": new_agent_text_message(
                        json.dumps(cancel_data, ensure_ascii=False),
                        context_id, task_id
                    ),
                },
                final=True,
                contextId=context_id,
                taskId=task_id,
                metadata={
                    "crew_type": "pdf2bpmn",
                    "event_type": "task_cancelled"
                }
            )
        )
        
        # HTTP 클라이언트 정리
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        
        logger.info("[CANCELLED] PDF2BPMN task cancelled")
