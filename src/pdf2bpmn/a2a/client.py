"""A2A client implementation for PDF2BPMN agent."""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, AsyncGenerator
from pathlib import Path

import httpx

from .protocol import (
    DiscoverResponse,
    ExecuteRequest,
    ExecuteResponse,
    TaskStatus,
    TaskResult,
)

logger = logging.getLogger(__name__)


class A2AClient:
    """A2A protocol client for PDF2BPMN agent."""
    
    def __init__(self, server_url: str = "http://localhost:9999"):
        """
        Initialize A2A client.
        
        Args:
            server_url: A2A server URL (default: http://localhost:9999)
        """
        self.server_url = server_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
    
    async def discover(self) -> DiscoverResponse:
        """Discover agent capabilities."""
        response = await self.client.get(f"{self.server_url}/discover")
        response.raise_for_status()
        return DiscoverResponse(**response.json())
    
    async def execute(
        self,
        pdf_url: Optional[str] = None,
        pdf_path: Optional[str] = None,
        pdf_file_name: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> ExecuteResponse:
        """
        Execute a PDF to BPMN conversion task.
        
        Args:
            pdf_url: URL of the PDF file
            pdf_path: Local file path to the PDF file
            pdf_file_name: Original PDF file name
            task_id: Optional task ID for idempotency
        
        Returns:
            ExecuteResponse with task_id
        """
        if not pdf_url and not pdf_path:
            raise ValueError("Either 'pdf_url' or 'pdf_path' must be provided")
        
        # If pdf_path is provided, convert to absolute path
        if pdf_path:
            pdf_path = str(Path(pdf_path).absolute())
            if not pdf_file_name:
                pdf_file_name = Path(pdf_path).name
        
        request = ExecuteRequest(
            input={
                "pdf_url": pdf_url,
                "pdf_path": pdf_path,
                "pdf_file_name": pdf_file_name
            },
            task_id=task_id
        )
        
        response = await self.client.post(
            f"{self.server_url}/execute",
            json=request.model_dump()
        )
        response.raise_for_status()
        return ExecuteResponse(**response.json())
    
    async def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        response = await self.client.get(f"{self.server_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_result(self, task_id: str) -> TaskResult:
        """Get task result."""
        response = await self.client.get(f"{self.server_url}/result/{task_id}")
        response.raise_for_status()
        return TaskResult(**response.json())
    
    async def cancel(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task."""
        response = await self.client.delete(f"{self.server_url}/task/{task_id}")
        response.raise_for_status()
        return response.json()
    
    async def stream_events(self, task_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream task events via Server-Sent Events.
        
        Args:
            task_id: Task identifier
        
        Yields:
            Event dictionaries
        """
        async with self.client.stream(
            "GET",
            f"{self.server_url}/events/{task_id}",
            headers={"Accept": "text/event-stream"}
        ) as response:
            response.raise_for_status()
            
            buffer = ""
            async for chunk in response.aiter_bytes():
                buffer += chunk.decode('utf-8')
                
                # Process complete lines
                while '\n\n' in buffer:
                    event_block, buffer = buffer.split('\n\n', 1)
                    event = self._parse_sse_event(event_block)
                    if event:
                        yield event
    
    def _parse_sse_event(self, event_block: str) -> Optional[Dict[str, Any]]:
        """Parse Server-Sent Event block."""
        lines = event_block.strip().split('\n')
        event = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'event':
                    event['event_type'] = value
                elif key == 'data':
                    import json
                    try:
                        event['data'] = json.loads(value)
                    except json.JSONDecodeError:
                        event['data'] = value
        
        return event if event else None
    
    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        show_progress: bool = True
    ) -> TaskResult:
        """
        Wait for task completion and return result.
        
        Args:
            task_id: Task identifier
            poll_interval: Polling interval in seconds
            show_progress: Whether to show progress updates
        
        Returns:
            TaskResult when completed
        """
        last_progress = -1
        
        while True:
            status = await self.get_status(task_id)
            current_status = status["status"]
            
            if show_progress and "result" in status and status["result"]:
                result = status["result"]
                if isinstance(result, dict):
                    progress = result.get("progress", 0)
                    message = result.get("message", "")
                    
                    if progress != last_progress:
                        print(f"[{progress}%] {message}")
                        last_progress = progress
            
            if current_status == TaskStatus.COMPLETED:
                return await self.get_result(task_id)
            elif current_status == TaskStatus.FAILED:
                result = await self.get_result(task_id)
                raise Exception(f"Task failed: {result.error}")
            elif current_status == TaskStatus.CANCELLED:
                raise Exception("Task was cancelled")
            
            await asyncio.sleep(poll_interval)
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()
