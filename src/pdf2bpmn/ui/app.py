"""Streamlit UI for PDF to BPMN converter."""
import streamlit as st
from pathlib import Path
import tempfile
import shutil

from ..config import Config
from ..workflow.graph import PDF2BPMNWorkflow, create_workflow, compile_workflow_with_checkpointer
from ..graph.neo4j_client import Neo4jClient
from ..models.entities import AmbiguityStatus


# Page configuration
st.set_page_config(
    page_title="PDF2BPMN Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono&display=swap');
    
    .stApp {
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .status-card.success {
        border-left: 4px solid #10b981;
    }
    
    .status-card.warning {
        border-left: 4px solid #f59e0b;
    }
    
    .status-card.error {
        border-left: 4px solid #ef4444;
    }
    
    .question-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #f59e0b;
    }
    
    .artifact-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    code {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = "upload"
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "neo4j_connected" not in st.session_state:
        st.session_state.neo4j_connected = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = None


def check_neo4j_connection():
    """Check if Neo4j is connected."""
    try:
        client = Neo4jClient()
        connected = client.verify_connection()
        client.close()
        return connected
    except Exception as e:
        st.error(f"Neo4j 연결 오류: {e}")
        return False


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>📄 PDF2BPMN Converter</h1>
        <p>업무 편람/정의서를 BPMN, DMN, Agent Skill 문서로 자동 변환</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with status and settings."""
    with st.sidebar:
        st.header("🔧 설정 및 상태")
        
        # Neo4j connection status
        st.subheader("📊 Neo4j 연결")
        if st.button("연결 확인"):
            if check_neo4j_connection():
                st.session_state.neo4j_connected = True
                st.success("✅ Neo4j 연결됨")
            else:
                st.session_state.neo4j_connected = False
                st.error("❌ Neo4j 연결 실패")
        
        if st.session_state.neo4j_connected:
            st.success("연결됨")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ 추출 설정")
        confidence_threshold = st.slider(
            "신뢰도 임계값",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="이 값 미만의 추출 결과는 사용자 확인이 필요합니다"
        )
        
        auto_merge = st.checkbox(
            "유사 엔티티 자동 병합",
            value=True,
            help="90% 이상 유사한 엔티티를 자동으로 병합합니다"
        )
        
        st.divider()
        
        # Processing status
        st.subheader("📈 처리 현황")
        if st.session_state.results:
            results = st.session_state.results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("프로세스", len(results.get("processes", [])))
                st.metric("역할", len(results.get("roles", [])))
            with col2:
                st.metric("태스크", len(results.get("tasks", [])))
                st.metric("스킬", len(results.get("skills", [])))
        
        st.divider()
        
        # Actions
        st.subheader("🔄 작업")
        if st.button("데이터베이스 초기화", type="secondary"):
            if st.session_state.neo4j_connected:
                client = Neo4jClient()
                client.init_schema()
                client.close()
                st.success("스키마 초기화 완료")


def render_upload_section():
    """Render the file upload section."""
    st.header("📁 PDF 업로드")
    
    uploaded_files = st.file_uploader(
        "업무 편람/정의서 PDF를 업로드하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="여러 파일을 동시에 업로드할 수 있습니다"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        st.subheader("📋 업로드된 파일")
        for file in uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {file.name}")
            with col2:
                st.text(f"{file.size / 1024:.1f} KB")
        
        return True
    return False


def save_uploaded_files(uploaded_files) -> list[str]:
    """Save uploaded files to disk and return paths."""
    paths = []
    Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    for file in uploaded_files:
        file_path = Config.UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(str(file_path))
    
    return paths


def render_processing_section():
    """Render the processing section."""
    st.header("⚙️ 처리 중...")
    
    if not st.session_state.uploaded_files:
        st.warning("먼저 PDF 파일을 업로드해주세요.")
        return
    
    # Save files
    with st.spinner("파일 저장 중..."):
        pdf_paths = save_uploaded_files(st.session_state.uploaded_files)
    
    # Initialize workflow
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create workflow
        workflow = PDF2BPMNWorkflow()
        
        # Initialize Neo4j schema
        status_text.text("Neo4j 스키마 초기화 중...")
        progress_bar.progress(10)
        workflow.neo4j.init_schema()
        
        # Process PDF
        status_text.text("PDF 분석 중...")
        progress_bar.progress(20)
        
        # Create initial state
        initial_state = {
            "pdf_paths": pdf_paths,
            "documents": [],
            "sections": [],
            "reference_chunks": [],
            "processes": [],
            "tasks": [],
            "roles": [],
            "gateways": [],
            "events": [],
            "skills": [],
            "dmn_decisions": [],
            "dmn_rules": [],
            "evidences": [],
            "open_questions": [],
            "resolved_questions": [],
            "current_question": None,
            "user_answer": None,
            "confidence_threshold": 0.8,
            "current_step": "ingest_pdf",
            "error": None,
            "bpmn_xml": None,
            "bpmn_xmls": {},
            "bpmn_files": {},
            "skill_docs": {},
            "dmn_xml": None
        }
        
        # Run workflow steps manually for better progress tracking
        status_text.text("📄 PDF 추출 중...")
        progress_bar.progress(30)
        state = workflow.ingest_pdf(initial_state)
        initial_state.update(state)
        
        status_text.text("📑 섹션 분석 중...")
        progress_bar.progress(40)
        state = workflow.segment_sections(initial_state)
        initial_state.update(state)
        
        status_text.text("🔍 엔티티 추출 중...")
        progress_bar.progress(50)
        state = workflow.extract_candidates(initial_state)
        initial_state.update(state)
        
        status_text.text("🔄 정규화 중...")
        progress_bar.progress(60)
        state = workflow.normalize_entities(initial_state)
        initial_state.update(state)
        
        status_text.text("❓ 모호성 검출 중...")
        progress_bar.progress(70)
        state = workflow.detect_ambiguities(initial_state)
        initial_state.update(state)
        
        # Check for questions
        open_questions = initial_state.get("open_questions", [])
        if open_questions:
            st.session_state.current_step = "questions"
            st.session_state.workflow_state = initial_state
            st.session_state.current_question = open_questions[0]
            progress_bar.progress(75)
            status_text.text("사용자 확인이 필요합니다...")
            st.rerun()
            return
        
        # Continue with generation
        status_text.text("📝 스킬 문서 생성 중...")
        progress_bar.progress(80)
        state = workflow.generate_skills(initial_state)
        initial_state.update(state)
        
        status_text.text("📊 DMN 생성 중...")
        progress_bar.progress(85)
        state = workflow.generate_dmn(initial_state)
        initial_state.update(state)
        
        status_text.text("🔧 BPMN 조립 중...")
        progress_bar.progress(90)
        state = workflow.assemble_bpmn(initial_state)
        initial_state.update(state)
        
        status_text.text("✔️ 검증 중...")
        progress_bar.progress(95)
        state = workflow.validate_consistency(initial_state)
        initial_state.update(state)
        
        status_text.text("📦 내보내기 중...")
        progress_bar.progress(100)
        state = workflow.export_artifacts(initial_state)
        initial_state.update(state)
        
        # Store results
        st.session_state.results = initial_state
        st.session_state.current_step = "results"
        
        status_text.text("✅ 완료!")
        st.success("처리가 완료되었습니다!")
        
    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_questions_section():
    """Render the HITL questions section."""
    st.header("🙋 확인이 필요한 항목")
    
    state = st.session_state.workflow_state
    if not state:
        st.warning("처리 상태를 찾을 수 없습니다.")
        return
    
    open_questions = [
        q for q in state.get("open_questions", [])
        if q.status == AmbiguityStatus.OPEN
    ]
    
    if not open_questions:
        st.session_state.current_step = "continue_processing"
        st.rerun()
        return
    
    current_q = open_questions[0]
    
    st.markdown(f"""
    <div class="question-card">
        <h3>❓ {current_q.question}</h3>
        <p><strong>관련 항목:</strong> {current_q.entity_type} (ID: {current_q.entity_id[:8]}...)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display options
    answer = st.radio(
        "답변을 선택하세요:",
        options=current_q.options,
        key=f"answer_{current_q.amb_id}"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("✅ 확인", type="primary"):
            # Apply answer
            current_q.status = AmbiguityStatus.RESOLVED
            current_q.answer = answer
            
            # Move to resolved
            resolved = state.get("resolved_questions", [])
            resolved.append(current_q)
            state["resolved_questions"] = resolved
            
            # Remove from open
            state["open_questions"] = [
                q for q in state["open_questions"]
                if q.amb_id != current_q.amb_id
            ]
            
            st.session_state.workflow_state = state
            st.rerun()
    
    with col2:
        if st.button("⏭️ 건너뛰기"):
            # Just move to next
            state["open_questions"] = [
                q for q in state["open_questions"]
                if q.amb_id != current_q.amb_id
            ]
            st.session_state.workflow_state = state
            st.rerun()
    
    # Progress indicator
    total = len(state.get("open_questions", [])) + len(state.get("resolved_questions", []))
    resolved_count = len(state.get("resolved_questions", []))
    st.progress(resolved_count / max(total, 1))
    st.text(f"진행률: {resolved_count}/{total}")


def render_continue_processing():
    """Continue processing after questions are answered."""
    st.header("⚙️ 처리 계속...")
    
    state = st.session_state.workflow_state
    workflow = PDF2BPMNWorkflow()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📝 스킬 문서 생성 중...")
        progress_bar.progress(25)
        result = workflow.generate_skills(state)
        state.update(result)
        
        status_text.text("📊 DMN 생성 중...")
        progress_bar.progress(50)
        result = workflow.generate_dmn(state)
        state.update(result)
        
        status_text.text("🔧 BPMN 조립 중...")
        progress_bar.progress(75)
        result = workflow.assemble_bpmn(state)
        state.update(result)
        
        status_text.text("✔️ 검증 중...")
        progress_bar.progress(90)
        result = workflow.validate_consistency(state)
        state.update(result)
        
        status_text.text("📦 내보내기 중...")
        progress_bar.progress(100)
        result = workflow.export_artifacts(state)
        state.update(result)
        
        st.session_state.results = state
        st.session_state.current_step = "results"
        st.success("✅ 처리가 완료되었습니다!")
        st.rerun()
        
    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")


def render_results_section():
    """Render the results section."""
    st.header("📊 처리 결과")
    
    results = st.session_state.results
    if not results:
        st.warning("결과를 찾을 수 없습니다.")
        return
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔄 프로세스",
            value=len(results.get("processes", []))
        )
    
    with col2:
        st.metric(
            label="📋 태스크",
            value=len(results.get("tasks", []))
        )
    
    with col3:
        st.metric(
            label="👤 역할",
            value=len(results.get("roles", []))
        )
    
    with col4:
        st.metric(
            label="🤖 스킬",
            value=len(results.get("skills", []))
        )
    
    st.divider()
    
    # Tabs for different outputs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 BPMN", "📊 DMN", "📝 스킬 문서", "🔍 엔티티"])
    
    with tab1:
        st.subheader("BPMN XML")
        bpmn_xml = results.get("bpmn_xml", "")
        if bpmn_xml:
            st.code(bpmn_xml[:3000] + "..." if len(bpmn_xml) > 3000 else bpmn_xml, language="xml")
            
            # Download button
            st.download_button(
                label="📥 BPMN 다운로드",
                data=bpmn_xml,
                file_name="process.bpmn",
                mime="application/xml"
            )
        else:
            st.info("BPMN이 생성되지 않았습니다.")
    
    with tab2:
        st.subheader("DMN 결정 테이블")
        dmn_xml = results.get("dmn_xml", "")
        if dmn_xml:
            st.code(dmn_xml[:2000] + "..." if len(dmn_xml) > 2000 else dmn_xml, language="xml")
            
            st.download_button(
                label="📥 DMN 다운로드",
                data=dmn_xml,
                file_name="decisions.dmn",
                mime="application/xml"
            )
        else:
            st.info("DMN 규칙이 추출되지 않았습니다.")
    
    with tab3:
        st.subheader("생성된 스킬 문서")
        skill_docs = results.get("skill_docs", {})
        
        if skill_docs:
            for task_id, markdown in skill_docs.items():
                with st.expander(f"📝 Skill: {task_id[:20]}..."):
                    st.markdown(markdown)
        else:
            st.info("스킬 문서가 생성되지 않았습니다 (에이전트 태스크 없음).")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("프로세스")
            for proc in results.get("processes", []):
                with st.expander(f"🔄 {proc.name}"):
                    st.text(f"ID: {proc.proc_id}")
                    st.text(f"목적: {proc.purpose or 'N/A'}")
            
            st.subheader("태스크")
            for task in results.get("tasks", []):
                with st.expander(f"📋 {task.name}"):
                    st.text(f"유형: {task.task_type.value}")
                    st.text(f"설명: {task.description or 'N/A'}")
        
        with col2:
            st.subheader("역할")
            for role in results.get("roles", []):
                with st.expander(f"👤 {role.name}"):
                    st.text(f"부서: {role.org_unit or 'N/A'}")
            
            st.subheader("게이트웨이")
            for gw in results.get("gateways", []):
                with st.expander(f"🔀 {gw.condition or gw.description}"):
                    st.text(f"유형: {gw.gateway_type.value}")
    
    st.divider()
    
    # Neo4j exploration link
    st.subheader("🔗 Neo4j 그래프 탐색")
    st.info("Neo4j Browser에서 그래프를 탐색하려면: http://localhost:7474")
    st.code("MATCH (n) RETURN n LIMIT 100", language="cypher")
    
    # Start over button
    if st.button("🔄 새로운 파일 처리", type="primary"):
        st.session_state.current_step = "upload"
        st.session_state.uploaded_files = []
        st.session_state.results = None
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content based on current step
    if st.session_state.current_step == "upload":
        if render_upload_section():
            if st.button("🚀 처리 시작", type="primary"):
                st.session_state.current_step = "processing"
                st.rerun()
    
    elif st.session_state.current_step == "processing":
        render_processing_section()
    
    elif st.session_state.current_step == "questions":
        render_questions_section()
    
    elif st.session_state.current_step == "continue_processing":
        render_continue_processing()
    
    elif st.session_state.current_step == "results":
        render_results_section()


if __name__ == "__main__":
    main()




