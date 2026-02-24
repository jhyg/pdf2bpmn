#!/usr/bin/env python3
"""
A2A Client for PDF2BPMN Agent
PDF 파일을 A2A Agent 서버(FastAPI)에 전송하여 BPMN을 추출하는 클라이언트

사용법:
    python a2a_client.py <pdf_path> [--server URL]

예시:
    python a2a_client.py ../docs/정수장\ 매뉴얼.pdf
    python a2a_client.py ../docs/정수장\ 매뉴얼.pdf --server http://localhost:8001
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path

import httpx


class PDF2BPMNClient:
    """A2A Client for communicating with the PDF2BPMN Agent Server."""
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    
    async def close(self):
        await self.client.aclose()
    
    async def health_check(self) -> dict:
        """서버 상태 확인"""
        resp = await self.client.get(f"{self.server_url}/api/health")
        resp.raise_for_status()
        return resp.json()
    
    async def check_neo4j(self) -> dict:
        """Neo4j 연결 상태 확인"""
        resp = await self.client.get(f"{self.server_url}/api/neo4j/status")
        resp.raise_for_status()
        return resp.json()
    
    async def upload_pdf(self, pdf_path: str) -> dict:
        """PDF 파일 업로드"""
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        with open(pdf_file, "rb") as f:
            files = {"file": (pdf_file.name, f, "application/pdf")}
            resp = await self.client.post(f"{self.server_url}/api/upload", files=files)
        
        resp.raise_for_status()
        return resp.json()
    
    async def start_processing(self, job_id: str) -> dict:
        """처리 시작"""
        resp = await self.client.post(f"{self.server_url}/api/process/{job_id}")
        resp.raise_for_status()
        return resp.json()
    
    async def get_job_status(self, job_id: str) -> dict:
        """작업 상태 조회"""
        resp = await self.client.get(f"{self.server_url}/api/jobs/{job_id}")
        resp.raise_for_status()
        return resp.json()
    
    async def poll_until_complete(self, job_id: str, poll_interval: float = 3.0) -> dict:
        """작업 완료까지 폴링"""
        start_time = time.time()
        last_progress = -1
        last_message = ""
        
        while True:
            status = await self.get_job_status(job_id)
            current_status = status.get("status", "unknown")
            progress = status.get("progress", 0)
            detail = status.get("detail_message", "")
            chunk_info = status.get("chunk_info")
            
            # 진행 상황 출력 (변경된 경우만)
            if progress != last_progress or detail != last_message:
                elapsed = time.time() - start_time
                progress_bar = "█" * (progress // 5) + "░" * (20 - progress // 5)
                
                chunk_str = ""
                if chunk_info:
                    chunk_str = f" [청크 {chunk_info.get('current', '?')}/{chunk_info.get('total', '?')}]"
                
                print(f"\r  [{progress_bar}] {progress:3d}% | {detail}{chunk_str} ({elapsed:.0f}s)", end="", flush=True)
                last_progress = progress
                last_message = detail
            
            if current_status == "completed":
                print()  # newline
                return status
            elif current_status == "error":
                print()
                raise RuntimeError(f"처리 중 오류 발생: {status.get('error', '알 수 없는 오류')}")
            
            await asyncio.sleep(poll_interval)
    
    async def get_processes(self) -> list:
        """추출된 프로세스 목록 조회"""
        resp = await self.client.get(f"{self.server_url}/api/processes")
        resp.raise_for_status()
        return resp.json()
    
    async def get_process_detail(self, proc_id: str) -> dict:
        """프로세스 상세 정보 조회"""
        resp = await self.client.get(f"{self.server_url}/api/processes/{proc_id}")
        resp.raise_for_status()
        return resp.json()
    
    async def get_tasks(self) -> list:
        """추출된 태스크 목록 조회"""
        resp = await self.client.get(f"{self.server_url}/api/tasks")
        resp.raise_for_status()
        return resp.json()
    
    async def get_bpmn_list(self) -> list:
        """BPMN 파일 목록 조회"""
        resp = await self.client.get(f"{self.server_url}/api/files/bpmn/list")
        resp.raise_for_status()
        return resp.json()
    
    async def get_bpmn_content(self, process_id: str = None) -> dict:
        """BPMN XML 내용 조회"""
        params = {}
        if process_id:
            params["process_id"] = process_id
        resp = await self.client.get(f"{self.server_url}/api/files/bpmn/content", params=params)
        resp.raise_for_status()
        return resp.json()
    
    async def get_all_bpmn(self) -> list:
        """모든 BPMN 내용 조회"""
        resp = await self.client.get(f"{self.server_url}/api/files/bpmn/all")
        resp.raise_for_status()
        return resp.json()
    
    async def get_graph_stats(self) -> dict:
        """Neo4j 그래프 통계 조회"""
        resp = await self.client.get(f"{self.server_url}/api/graph-stats")
        resp.raise_for_status()
        return resp.json()


async def main():
    parser = argparse.ArgumentParser(
        description="A2A Client - PDF에서 BPMN 프로세스 추출"
    )
    parser.add_argument("pdf_path", help="변환할 PDF 파일 경로")
    parser.add_argument("--server", default="http://localhost:8001", help="A2A 서버 URL (기본: http://localhost:8001)")
    parser.add_argument("--output-dir", default="./output_bpmn", help="BPMN 출력 디렉토리")
    parser.add_argument("--skip-clear", action="store_true", help="Neo4j 초기화 건너뛰기")
    
    args = parser.parse_args()
    
    client = PDF2BPMNClient(args.server)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("=" * 70)
        print("🤖 A2A Client - PDF to BPMN Process Extractor")
        print("=" * 70)
        print(f"📄 PDF: {args.pdf_path}")
        print(f"🌐 Server: {args.server}")
        print(f"📁 Output: {output_dir}")
        print()
        
        # 1. 서버 상태 확인
        print("🔍 Step 1: 서버 상태 확인...")
        try:
            health = await client.health_check()
            print(f"   ✅ 서버 정상: {health}")
        except Exception as e:
            print(f"   ❌ 서버 연결 실패: {e}")
            print(f"   ℹ️  서버가 실행 중인지 확인하세요: python run.py api --port 8001")
            return 1
        
        # 2. Neo4j 확인
        print("\n🔍 Step 2: Neo4j 연결 확인...")
        try:
            neo4j_status = await client.check_neo4j()
            print(f"   ✅ Neo4j: {neo4j_status}")
        except Exception as e:
            print(f"   ⚠️  Neo4j 상태 확인 실패: {e}")
            print(f"   ℹ️  Neo4j가 실행 중이어야 합니다 (bolt://localhost:7687)")
        
        # 3. Neo4j 초기화 (선택적)
        if not args.skip_clear:
            print("\n🧹 Step 3: Neo4j 데이터 초기화...")
            try:
                resp = await client.client.post(f"{client.server_url}/api/neo4j/clear")
                if resp.status_code == 200:
                    print(f"   ✅ Neo4j 초기화 완료: {resp.json()}")
                else:
                    print(f"   ⚠️  Neo4j 초기화 실패: {resp.text}")
            except Exception as e:
                print(f"   ⚠️  Neo4j 초기화 건너뜀: {e}")
        
        # 4. PDF 업로드
        print(f"\n📤 Step 4: PDF 파일 업로드...")
        upload_result = await client.upload_pdf(args.pdf_path)
        job_id = upload_result["job_id"]
        print(f"   ✅ 업로드 완료!")
        print(f"   📋 Job ID: {job_id}")
        print(f"   📄 File: {upload_result.get('file_name', 'N/A')}")
        
        # 5. 처리 시작
        print(f"\n⚙️  Step 5: BPMN 추출 처리 시작...")
        process_result = await client.start_processing(job_id)
        print(f"   ✅ {process_result.get('message', 'Processing started')}")
        
        # 6. 완료 대기 (폴링)
        print(f"\n⏳ Step 6: 처리 진행 중 (실시간 모니터링)...")
        final_status = await client.poll_until_complete(job_id, poll_interval=3.0)
        
        result = final_status.get("result", {})
        print(f"\n   🎉 처리 완료!")
        print(f"   📊 결과 요약:")
        print(f"      - 프로세스: {result.get('processes', 0)}개")
        print(f"      - 태스크: {result.get('tasks', 0)}개")
        print(f"      - 역할: {result.get('roles', 0)}개")
        print(f"      - 게이트웨이: {result.get('gateways', 0)}개")
        print(f"      - 의사결정: {result.get('decisions', 0)}개")
        
        # 7. 프로세스 목록 조회
        print(f"\n📋 Step 7: 추출된 프로세스 목록...")
        try:
            processes = await client.get_processes()
            if processes:
                for i, proc in enumerate(processes, 1):
                    proc_id = proc.get("proc_id", proc.get("id", "N/A"))
                    proc_name = proc.get("name", "N/A")
                    task_count = proc.get("task_count", "?")
                    print(f"   {i}. [{proc_id[:8]}...] {proc_name} (태스크: {task_count})")
            else:
                print("   (프로세스 없음)")
        except Exception as e:
            print(f"   ⚠️  프로세스 목록 조회 실패: {e}")
        
        # 8. BPMN XML 가져오기 및 저장
        print(f"\n💾 Step 8: BPMN XML 저장...")
        try:
            bpmn_list = await client.get_all_bpmn()
            if bpmn_list:
                for bpmn_item in bpmn_list:
                    proc_id = bpmn_item.get("process_id", "unknown")
                    proc_name = bpmn_item.get("process_name", "process")
                    bpmn_xml = bpmn_item.get("content", "")
                    
                    # 파일명 생성
                    safe_name = proc_name.replace(" ", "_").replace("/", "_")[:50]
                    filename = f"{safe_name}_{proc_id[:8]}.bpmn"
                    filepath = output_dir / filename
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(bpmn_xml)
                    
                    print(f"   ✅ {filename} ({len(bpmn_xml)} bytes)")
                
                print(f"\n   📁 모든 BPMN 파일이 {output_dir}에 저장되었습니다.")
            else:
                print("   ⚠️  생성된 BPMN이 없습니다.")
        except Exception as e:
            print(f"   ⚠️  BPMN 저장 실패: {e}")
        
        # 9. 태스크 상세 정보
        print(f"\n📝 Step 9: 추출된 태스크 상세 정보...")
        try:
            tasks = await client.get_tasks()
            if tasks:
                for i, task in enumerate(tasks[:20], 1):  # 최대 20개만 표시
                    task_name = task.get("name", "N/A")
                    task_type = task.get("task_type", "N/A")
                    role = task.get("role_name", task.get("role", "N/A"))
                    process_name = task.get("process_name", "")
                    print(f"   {i:2d}. [{task_type:12s}] {task_name}")
                    if role != "N/A":
                        print(f"       └─ 담당: {role} | 프로세스: {process_name}")
                
                if len(tasks) > 20:
                    print(f"   ... 외 {len(tasks) - 20}개 태스크")
            else:
                print("   (태스크 없음)")
        except Exception as e:
            print(f"   ⚠️  태스크 조회 실패: {e}")
        
        # 10. 그래프 통계
        print(f"\n📊 Step 10: Neo4j 그래프 통계...")
        try:
            stats = await client.get_graph_stats()
            print(f"   - 전체 노드: {stats.get('total_nodes', 'N/A')}")
            print(f"   - 전체 관계: {stats.get('total_relationships', 'N/A')}")
            node_counts = stats.get("node_counts", {})
            if node_counts:
                print(f"   - 노드 타입별:")
                for label, count in node_counts.items():
                    print(f"     · {label}: {count}")
        except Exception as e:
            print(f"   ⚠️  그래프 통계 조회 실패: {e}")
        
        print()
        print("=" * 70)
        print("✅ A2A PDF-to-BPMN 변환 완료!")
        print("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일 오류: {e}")
        return 1
    except RuntimeError as e:
        print(f"\n❌ 처리 오류: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
