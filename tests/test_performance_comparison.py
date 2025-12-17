"""
성능 비교 테스트

다양한 최적화 옵션을 적용하여 속도를 측정하고 비교합니다.

테스트 시나리오:
1. 기본 설정 (chunk_size=1000, overlap=200, evidence=full)
2. 출처 연결 off (evidence=off)
3. 큰 청크 크기 (chunk_size=4000, overlap=100)
4. 시맨틱 청킹 (semantic chunking)
5. 조합: 큰 청크 + 출처 off
"""

import sys
import time
import os
from pathlib import Path
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf2bpmn.extractors.pdf_extractor import PDFExtractor
from pdf2bpmn.extractors.entity_extractor import EntityExtractor
from pdf2bpmn.graph.neo4j_client import Neo4jClient
from pdf2bpmn.workflow.graph import PDF2BPMNWorkflow
from pdf2bpmn.config import Config


@contextmanager
def timer(name: str):
    """시간 측정 컨텍스트 매니저"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"   ⏱️ [{name}] {elapsed:.2f}초")


class PerformanceTest:
    """성능 비교 테스트"""
    
    def __init__(self):
        self.results = []
    
    def setup(self):
        """초기화"""
        self.neo4j = Neo4jClient()
        self._clear_neo4j()
        self.neo4j.init_schema()
    
    def teardown(self):
        """정리"""
        if self.neo4j:
            self.neo4j.close()
    
    def _clear_neo4j(self):
        """Neo4j 데이터베이스 초기화"""
        with self.neo4j.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def run_test(self, name: str, config: dict, pdf_path: str):
        """특정 설정으로 테스트 실행"""
        print(f"\n{'='*70}")
        print(f"🧪 테스트: {name}")
        print(f"{'='*70}")
        print(f"설정: {config}")
        
        # 설정 적용
        original_chunk_size = Config.CHUNK_SIZE
        original_chunk_overlap = Config.CHUNK_OVERLAP
        original_evidence_mode = Config.EVIDENCE_MODE
        original_chunking_strategy = Config.CHUNKING_STRATEGY
        
        Config.CHUNK_SIZE = config.get("chunk_size", Config.CHUNK_SIZE)
        Config.CHUNK_OVERLAP = config.get("chunk_overlap", Config.CHUNK_OVERLAP)
        Config.EVIDENCE_MODE = config.get("evidence_mode", Config.EVIDENCE_MODE)
        Config.CHUNKING_STRATEGY = config.get("chunking_strategy", Config.CHUNKING_STRATEGY)
        
        try:
            # Neo4j 초기화
            self._clear_neo4j()
            
            test_start = time.time()
            timings = {}
            
            # PDF 추출
            with timer("PDF 추출"):
                pdf_extractor = PDFExtractor(
                    chunk_size=config.get("chunk_size"),
                    chunk_overlap=config.get("chunk_overlap"),
                    chunking_strategy=config.get("chunking_strategy")
                )
                document, sections, chunks = pdf_extractor.extract_document(pdf_path)
                timings["pdf_extraction"] = time.time() - test_start
            
            print(f"   📄 문서: {document.title}")
            print(f"   📑 페이지: {document.page_count}")
            print(f"   📋 청크 수: {len(chunks)}")
            print(f"   📊 섹션 수: {len(sections)}")
            
            # 청크 평균 크기
            if chunks:
                avg_chunk_size = sum(len(c.text) for c in chunks) / len(chunks)
                print(f"   📏 평균 청크 크기: {avg_chunk_size:.0f}자")
            
            # 워크플로우 실행
            workflow = PDF2BPMNWorkflow()
            
            state = {
                "pdf_paths": [pdf_path],
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
                "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                "current_step": "ingest_pdf",
                "error": None,
                "bpmn_xml": None,
                "bpmn_xmls": {},
                "bpmn_files": {},
                "skill_docs": {},
                "dmn_xml": None
            }
            
            # Step 1: Ingest PDF
            step_start = time.time()
            result = workflow.ingest_pdf(state)
            state.update(result)
            timings["ingest_pdf"] = time.time() - step_start
            
            # Step 2: Segment sections
            step_start = time.time()
            result = workflow.segment_sections(state)
            state.update(result)
            timings["segment_sections"] = time.time() - step_start
            
            # Step 3: Extract candidates
            step_start = time.time()
            result = workflow.extract_candidates(state)
            state.update(result)
            timings["extract_candidates"] = time.time() - step_start
            
            # Step 4: Normalize
            step_start = time.time()
            result = workflow.normalize_entities(state)
            state.update(result)
            timings["normalize_entities"] = time.time() - step_start
            
            # Step 5: Generate skills
            step_start = time.time()
            result = workflow.generate_skills(state)
            state.update(result)
            timings["generate_skills"] = time.time() - step_start
            
            # Step 6: Generate DMN
            step_start = time.time()
            result = workflow.generate_dmn(state)
            state.update(result)
            timings["generate_dmn"] = time.time() - step_start
            
            # Step 7: Assemble BPMN
            step_start = time.time()
            result = workflow.assemble_bpmn(state)
            state.update(result)
            timings["assemble_bpmn"] = time.time() - step_start
            
            # Step 8: Export
            step_start = time.time()
            result = workflow.export_artifacts(state)
            state.update(result)
            timings["export_artifacts"] = time.time() - step_start
            
            total_time = time.time() - test_start
            timings["total"] = total_time
            
            # 결과 수집
            process_count = len(state.get("processes", []))
            task_count = len(state.get("tasks", []))
            role_count = len(state.get("roles", []))
            
            print(f"\n📊 결과:")
            print(f"   프로세스: {process_count}개")
            print(f"   태스크: {task_count}개")
            print(f"   역할: {role_count}개")
            print(f"   총 소요시간: {total_time:.2f}초")
            
            result_data = {
                "name": name,
                "config": config,
                "chunk_count": len(chunks),
                "avg_chunk_size": avg_chunk_size if chunks else 0,
                "process_count": process_count,
                "task_count": task_count,
                "role_count": role_count,
                "timings": timings
            }
            
            self.results.append(result_data)
            workflow.neo4j.close()
            
            return result_data
            
        finally:
            # 원래 설정 복원
            Config.CHUNK_SIZE = original_chunk_size
            Config.CHUNK_OVERLAP = original_chunk_overlap
            Config.EVIDENCE_MODE = original_evidence_mode
            Config.CHUNKING_STRATEGY = original_chunking_strategy
    
    def print_comparison(self):
        """결과 비교 출력"""
        if not self.results:
            return
        
        print(f"\n{'='*70}")
        print("📊 성능 비교 결과")
        print(f"{'='*70}")
        
        # 헤더
        print(f"\n{'테스트':<30} {'청크수':<8} {'평균청크':<10} {'총시간':<10} {'PDF':<8} {'추출':<8} {'정규화':<8}")
        print("-" * 70)
        
        baseline = self.results[0] if self.results else None
        
        for result in self.results:
            timings = result["timings"]
            name = result["name"][:28]
            chunk_count = result["chunk_count"]
            avg_chunk = f"{result['avg_chunk_size']:.0f}"
            total = f"{timings['total']:.2f}"
            pdf_time = f"{timings.get('pdf_extraction', 0):.2f}"
            extract_time = f"{timings.get('extract_candidates', 0):.2f}"
            normalize_time = f"{timings.get('normalize_entities', 0):.2f}"
            
            # 개선율 계산
            if baseline and baseline["timings"]["total"] > 0:
                improvement = ((baseline["timings"]["total"] - timings["total"]) / baseline["timings"]["total"]) * 100
                improvement_str = f" ({improvement:+.1f}%)" if improvement != 0 else ""
            else:
                improvement_str = ""
            
            print(f"{name:<30} {chunk_count:<8} {avg_chunk:<10} {total:<10}{improvement_str} {pdf_time:<8} {extract_time:<8} {normalize_time:<8}")
        
        print(f"\n{'='*70}")
        print("상세 시간 분석")
        print(f"{'='*70}")
        
        for result in self.results:
            print(f"\n{result['name']}:")
            timings = result["timings"]
            for step, duration in sorted(timings.items(), key=lambda x: x[1], reverse=True):
                if step != "total":
                    percentage = (duration / timings["total"]) * 100
                    print(f"   {step:<20}: {duration:>6.2f}초 ({percentage:>5.1f}%)")


def main():
    """메인 테스트 실행"""
    # 여러 경로에서 PDF 파일 찾기
    possible_paths = [
        Path(__file__).parent.parent / "doc" / "purchase_approval_process.pdf",
        Path(__file__).parent.parent / "uploads" / "purchase_approval_process.pdf",
    ]
    
    pdf_path = None
    for path in possible_paths:
        if path.exists():
            pdf_path = path
            break
    
    if not pdf_path:
        print(f"❌ PDF 파일을 찾을 수 없습니다.")
        print("   다음 경로를 확인했습니다:")
        for path in possible_paths:
            print(f"     - {path}")
        return
    
    test = PerformanceTest()
    test.setup()
    
    try:
        # 테스트 1: 기본 설정
        test.run_test(
            "1. 기본 설정",
            {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "evidence_mode": "full",
                "chunking_strategy": "fixed"
            },
            str(pdf_path)
        )
        
        # 테스트 2: 출처 연결 off
        test.run_test(
            "2. 출처 연결 OFF",
            {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "evidence_mode": "off",
                "chunking_strategy": "fixed"
            },
            str(pdf_path)
        )
        
        # 테스트 3: 큰 청크 크기
        test.run_test(
            "3. 큰 청크 (4000자, 오버랩 100)",
            {
                "chunk_size": 4000,
                "chunk_overlap": 100,
                "evidence_mode": "full",
                "chunking_strategy": "fixed"
            },
            str(pdf_path)
        )
        
        # 테스트 4: 시맨틱 청킹
        test.run_test(
            "4. 시맨틱 청킹",
            {
                "chunk_size": 1000,
                "chunk_overlap": 50,
                "evidence_mode": "full",
                "chunking_strategy": "semantic"
            },
            str(pdf_path)
        )
        
        # 테스트 5: 조합 (큰 청크 + 출처 off)
        test.run_test(
            "5. 조합 (큰 청크 + 출처 OFF)",
            {
                "chunk_size": 4000,
                "chunk_overlap": 100,
                "evidence_mode": "off",
                "chunking_strategy": "fixed"
            },
            str(pdf_path)
        )
        
        # 결과 비교
        test.print_comparison()
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.teardown()


if __name__ == "__main__":
    main()

