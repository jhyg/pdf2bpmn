"""
테스트: PDF → Neo4j → BPMN 전체 파이프라인

이 테스트는 실제 PDF 파일을 입력으로 받아
전체 파이프라인을 실행하고 BPMN을 생성합니다.

테스트 파일: uploads/purchase_approval_process.pdf
"""

import sys
import time
from pathlib import Path
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf2bpmn.extractors.pdf_extractor import PDFExtractor
from pdf2bpmn.extractors.entity_extractor import EntityExtractor
from pdf2bpmn.graph.neo4j_client import Neo4jClient
from pdf2bpmn.generators.bpmn_generator import BPMNGenerator
from pdf2bpmn.models.entities import generate_id, Process, Task, Role, Gateway
from pdf2bpmn.workflow.graph import PDF2BPMNWorkflow


@contextmanager
def timer(name: str):
    """시간 측정 컨텍스트 매니저"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"   ⏱️ [{name}] {elapsed:.2f}초")


class TestPDFToBPMN:
    """PDF → BPMN 전체 파이프라인 테스트"""
    
    def __init__(self):
        self.neo4j = None
        self.pdf_extractor = None
        self.entity_extractor = None
        
    def setup(self):
        """초기화"""
        print("\n🔧 Setup 시작...")
        setup_start = time.time()
        
        with timer("Neo4jClient 생성"):
            self.neo4j = Neo4jClient()
        
        with timer("PDFExtractor 생성"):
            self.pdf_extractor = PDFExtractor()
        
        with timer("EntityExtractor 생성"):
            self.entity_extractor = EntityExtractor()
        
        # Neo4j 데이터 초기화
        with timer("Neo4j 초기화"):
            self._clear_neo4j()
            self._init_schema()
        
        print(f"   ⏱️ [Setup 총] {time.time() - setup_start:.2f}초")
    
    def teardown(self):
        """정리"""
        if self.neo4j:
            self.neo4j.close()
    
    def _clear_neo4j(self):
        """Neo4j 데이터베이스 초기화"""
        with self.neo4j.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("   ✅ Neo4j 초기화 완료")
    
    def _init_schema(self):
        """스키마 초기화"""
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT proc_id IF NOT EXISTS FOR (p:Process) REQUIRE p.proc_id IS UNIQUE",
            "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE",
            "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (r:Role) REQUIRE r.role_id IS UNIQUE",
            "CREATE CONSTRAINT gateway_id IF NOT EXISTS FOR (g:Gateway) REQUIRE g.gateway_id IS UNIQUE",
        ]
        
        with self.neo4j.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass
    
    def test_full_pipeline(self, pdf_path: str = None):
        """
        테스트: PDF → 추출 → Neo4j → BPMN 전체 파이프라인
        """
        print("\n" + "="*70)
        print("🚀 PDF → BPMN 전체 파이프라인 테스트")
        print("="*70)
        
        test_start = time.time()
        
        # PDF 경로 설정
        if pdf_path is None:
            pdf_path = Path(__file__).parent.parent / "uploads" / "purchase_approval_process.pdf"
        else:
            pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
            print("   먼저 scripts/create_sample_pdf.py를 실행하세요.")
            return None
        
        print(f"\n📄 입력 PDF: {pdf_path}")
        print(f"   파일 크기: {pdf_path.stat().st_size:,} bytes")
        
        # ========================================
        # Step 1: PDF 텍스트 추출 및 청킹
        # ========================================
        print("\n" + "-"*60)
        print("📖 Step 1: PDF 텍스트 추출 및 청킹")
        print("-"*60)
        
        with timer("PDF 추출"):
            document, sections, chunks = self.pdf_extractor.extract_document(str(pdf_path))
        
        doc_id = document.doc_id
        
        print(f"   문서 제목: {document.title or 'N/A'}")
        print(f"   페이지 수: {document.page_count}")
        print(f"   섹션 수: {len(sections)}")
        print(f"   청크 수: {len(chunks)}")
        
        # 청크 미리보기
        if chunks:
            print(f"\n   📋 청크 미리보기 (처음 3개):")
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk.text[:100].replace('\n', ' ')
                print(f"      {i+1}. [{len(chunk.text)}자] {preview}...")
        
        # ========================================
        # Step 2: 엔티티 추출 (LLM)
        # ========================================
        print("\n" + "-"*60)
        print("🤖 Step 2: 엔티티 추출 (LLM)")
        print("-"*60)
        
        all_processes = []
        all_tasks = []
        all_roles = []
        all_gateways = []
        all_events = []
        
        process_name_to_id = {}
        role_name_to_id = {}
        task_role_map = {}
        task_process_map = {}
        sequence_flows = []
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            print(f"\n   📄 Chunk {i+1}/{len(chunks)} 처리 중...")
            
            # 기존 컨텍스트
            existing_process_names = list(process_name_to_id.keys())
            existing_role_names = list(role_name_to_id.keys())
            
            if existing_process_names:
                print(f"      [컨텍스트] 기존 프로세스: {existing_process_names[:3]}...")
            
            # LLM 추출
            with timer(f"Chunk {i+1} LLM 추출"):
                extracted = self.entity_extractor.extract_from_text(
                    chunk.text,
                    existing_processes=existing_process_names,
                    existing_roles=existing_role_names
                )
            
            # 엔티티 변환
            entities = self.entity_extractor.convert_to_entities(
                extracted,
                doc_id,
                chunk_id=chunk.chunk_id,
                existing_processes=process_name_to_id,
                existing_roles=role_name_to_id
            )
            
            # 수집
            all_processes.extend(entities["processes"])
            all_tasks.extend(entities["tasks"])
            all_roles.extend(entities["roles"])
            all_gateways.extend(entities["gateways"])
            all_events.extend(entities.get("events", []))
            
            # 매핑 업데이트
            for proc in entities["processes"]:
                process_name_to_id[proc.name.lower()] = proc.proc_id
            
            for role in entities["roles"]:
                role_name_to_id[role.name.lower()] = role.role_id
            
            # Task-Role, Task-Process 매핑
            if entities.get("task_role_map"):
                task_role_map.update(entities["task_role_map"])
            
            if entities.get("task_process_map"):
                task_process_map.update(entities["task_process_map"])
            else:
                for task in entities["tasks"]:
                    if task.process_id:
                        task_process_map[task.task_id] = task.process_id
            
            # 시퀀스 플로우
            if entities.get("sequence_flows"):
                sequence_flows.extend(entities["sequence_flows"])
            
            print(f"      추출: 프로세스 {len(entities['processes'])}, "
                  f"태스크 {len(entities['tasks'])}, "
                  f"역할 {len(entities['roles'])}, "
                  f"게이트웨이 {len(entities['gateways'])}")
            print(f"      ⏱️ [Chunk {i+1} 총] {time.time() - chunk_start:.2f}초")
        
        # 결과 요약
        print(f"\n   📊 추출 결과 요약:")
        print(f"      프로세스: {len(all_processes)}개")
        print(f"      태스크: {len(all_tasks)}개")
        print(f"      역할: {len(all_roles)}개")
        print(f"      게이트웨이: {len(all_gateways)}개")
        print(f"      시퀀스 플로우: {len(sequence_flows)}개")
        
        # ========================================
        # Step 3: 프로세스 병합 (동일 이름 통합)
        # ========================================
        print("\n" + "-"*60)
        print("🔗 Step 3: 프로세스 병합")
        print("-"*60)
        
        merged_processes, process_id_mapping = self._merge_duplicate_processes(all_processes)
        
        print(f"   병합 전: {len(all_processes)}개 → 병합 후: {len(merged_processes)}개")
        
        if process_id_mapping:
            print(f"   ID 매핑: {len(process_id_mapping)}개")
            
            # 태스크의 process_id 업데이트
            for task in all_tasks:
                if task.process_id in process_id_mapping:
                    task.process_id = process_id_mapping[task.process_id]
        
        # ========================================
        # Step 4: Neo4j 저장
        # ========================================
        print("\n" + "-"*60)
        print("💾 Step 4: Neo4j 저장")
        print("-"*60)
        
        with timer("Neo4j 저장"):
            # 프로세스 저장
            for proc in merged_processes:
                self.neo4j.create_process(proc)
            print(f"   프로세스 저장: {len(merged_processes)}개")
            
            # 역할 저장 (중복 제거)
            unique_roles = {}
            for role in all_roles:
                if role.name.lower() not in unique_roles:
                    unique_roles[role.name.lower()] = role
            
            for role in unique_roles.values():
                self.neo4j.create_role(role)
            print(f"   역할 저장: {len(unique_roles)}개")
            
            # 태스크 저장
            for task in all_tasks:
                self.neo4j.create_task(task)
            print(f"   태스크 저장: {len(all_tasks)}개")
            
            # 게이트웨이 저장
            for gw in all_gateways:
                self.neo4j.create_gateway(gw)
            print(f"   게이트웨이 저장: {len(all_gateways)}개")
            
            # 관계 생성
            self._create_relationships(merged_processes, all_tasks, all_gateways, 
                                       task_role_map, task_process_map)
            
            # 시퀀스 플로우 생성
            self._create_sequence_flows(merged_processes)
        
        # Neo4j 저장 확인
        print("\n   📊 Neo4j 저장 확인:")
        with self.neo4j.session() as session:
            result = session.run("MATCH (p:Process) RETURN count(p) as count")
            print(f"      Process: {result.single()['count']}개")
            
            result = session.run("MATCH (t:Task) RETURN count(t) as count")
            print(f"      Task: {result.single()['count']}개")
            
            result = session.run("MATCH (r:Role) RETURN count(r) as count")
            print(f"      Role: {result.single()['count']}개")
            
            result = session.run("MATCH ()-[r:NEXT]->() RETURN count(r) as count")
            next_count = result.single()['count']
            print(f"      NEXT 관계: {next_count}개")
        
        # ========================================
        # Step 5: BPMN 생성
        # ========================================
        print("\n" + "-"*60)
        print("📐 Step 5: BPMN 생성")
        print("-"*60)
        
        if not merged_processes:
            print("   ❌ 프로세스가 없어 BPMN을 생성할 수 없습니다.")
            return None
        
        main_process = merged_processes[0]
        print(f"   대상 프로세스: {main_process.name}")
        
        # 해당 프로세스의 태스크
        process_tasks = [t for t in all_tasks if t.process_id == main_process.proc_id]
        print(f"   태스크 수: {len(process_tasks)}개")
        
        with timer("BPMN 생성"):
            generator = BPMNGenerator()
            bpmn_xml = generator.generate(
                process=main_process,
                tasks=process_tasks,
                roles=list(unique_roles.values()),
                gateways=[g for g in all_gateways if g.process_id == main_process.proc_id],
                events=[],
                task_role_map=task_role_map
            )
        
        # 파일 저장
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        safe_name = pdf_path.stem
        output_file = output_dir / f"{safe_name}_pipeline.bpmn"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(bpmn_xml)
        
        print(f"\n   💾 BPMN 저장: {output_file}")
        print(f"      파일 크기: {len(bpmn_xml):,} bytes")
        print(f"      라인 수: {len(bpmn_xml.splitlines())}줄")
        
        # XML 검증
        print(f"\n   ✅ XML 검증:")
        print(f"      definitions: {'있음' if '<bpmn:definitions' in bpmn_xml else '없음'}")
        print(f"      process: {'있음' if '<bpmn:process' in bpmn_xml else '없음'}")
        print(f"      태스크: {bpmn_xml.count('<bpmn:userTask') + bpmn_xml.count('<bpmn:task ')}개")
        print(f"      시퀀스 플로우: {bpmn_xml.count('<bpmn:sequenceFlow')}개")
        
        # ========================================
        # 완료
        # ========================================
        print("\n" + "="*70)
        total_time = time.time() - test_start
        print(f"✅ 전체 파이프라인 완료! 총 소요시간: {total_time:.2f}초")
        print("="*70)
        
        return {
            "pdf_path": str(pdf_path),
            "bpmn_path": str(output_file),
            "processes": merged_processes,
            "tasks": all_tasks,
            "roles": list(unique_roles.values()),
            "gateways": all_gateways,
            "total_time": total_time
        }
    
    def _merge_duplicate_processes(self, processes: list) -> tuple:
        """같은 이름의 프로세스 병합"""
        name_to_processes = {}
        for proc in processes:
            name_key = proc.name.lower().strip()
            if name_key not in name_to_processes:
                name_to_processes[name_key] = []
            name_to_processes[name_key].append(proc)
        
        merged_processes = []
        process_id_mapping = {}
        
        for name_key, proc_group in name_to_processes.items():
            primary = proc_group[0]
            for other in proc_group[1:]:
                process_id_mapping[other.proc_id] = primary.proc_id
                if other.description and other.description not in (primary.description or ""):
                    primary.description = (primary.description or "") + " " + other.description
            merged_processes.append(primary)
        
        return merged_processes, process_id_mapping
    
    def _create_relationships(self, processes, tasks, gateways, task_role_map, task_process_map):
        """관계 생성"""
        # Process-Task 관계는 create_task에서 이미 생성됨
        
        # Task-Role 관계
        for task_id, role_id in task_role_map.items():
            try:
                self.neo4j.link_task_to_role(task_id, role_id)
            except Exception as e:
                pass
        
        print(f"   Task-Role 연결: {len(task_role_map)}개")
    
    def _create_sequence_flows(self, processes: list):
        """태스크 간 시퀀스 플로우 생성 (order 기준)"""
        created = 0
        for proc in processes:
            try:
                self.neo4j.create_task_sequence_for_process(proc.proc_id)
                created += 1
            except Exception as e:
                print(f"   시퀀스 생성 실패: {e}")
        
        print(f"   시퀀스 플로우 생성: {created}개 프로세스")


if __name__ == "__main__":
    import sys
    
    test = TestPDFToBPMN()
    
    # 커맨드 라인에서 PDF 경로 지정 가능
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        test.setup()
        result = test.test_full_pipeline(pdf_path)
        
        if result:
            print(f"\n📋 결과 요약:")
            print(f"   입력: {result['pdf_path']}")
            print(f"   출력: {result['bpmn_path']}")
            print(f"   프로세스: {len(result['processes'])}개")
            print(f"   태스크: {len(result['tasks'])}개")
            print(f"   역할: {len(result['roles'])}개")
            
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.teardown()

