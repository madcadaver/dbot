# tinygen_controller.py

import os
import logging
import json
import httpx
from typing import List, Optional, Dict, Any

dev_logger = logging.getLogger('dev')

class TinyGenController:
    """
    Acts as the exclusive API client for the TinyGen service.
    Handles storing knowledge, managing the scratchpad, searching,
    and controlling the background processing queue.
    """
    def __init__(self):
        self.api_url = os.getenv("TINYGEN_API_URL")
        if not self.api_url:
            dev_logger.error("TINYGEN_API_URL is not configured. TinyGenController will be disabled.")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data_body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """A generic helper to make HTTP requests to the TinyGen API."""
        if not self.api_url:
            dev_logger.error(f"Cannot make request, TINYGEN_API_URL is not set.")
            return None

        url = f"{self.api_url.rstrip('/')}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    content=data_body,
                    headers=headers
                )
                response.raise_for_status()
                dev_logger.info(f"Successfully sent '{method}' to TinyGen endpoint '{endpoint}'. Status: {response.status_code}")
                return response.json()
        except httpx.RequestError as e:
            dev_logger.error(f"Could not connect to TinyGen endpoint {url}: {e}")
        except httpx.HTTPStatusError as e:
            dev_logger.error(f"HTTP error for TinyGen endpoint {url}: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            dev_logger.error(f"An unexpected error occurred while calling TinyGen endpoint {url}: {e}", exc_info=True)
        return None

    # --- Knowledge Storage ---
    async def store_knowledge(self, unstructured_text: str, author_ref: Optional[str] = None, subject_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Sends unstructured text to TinyGen for processing and storage."""
        params = {}
        if author_ref:
            params['author_ref'] = author_ref
        if subject_hint:
            params['subject_hint'] = subject_hint

        return await self._make_request(
            method='POST',
            endpoint='/store',
            params=params,
            data_body=unstructured_text.encode('utf-8'),
            headers={'Content-Type': 'text/plain'}
        )

    # --- Scratchpad Management ---
    async def add_to_scratchpad(self, content: str, author_ref: str, tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Adds a new entry to Gen's private scratchpad."""
        json_payload = {"content": content, "tags": tags or []}
        params = {"author_ref": author_ref}
        return await self._make_request(
            method='POST',
            endpoint='/scratchpad/add',
            params=params,
            json_body=json_payload,
            headers={'Content-Type': 'application/json'}
        )

    async def search_scratchpad(self, query: str, author_ref: str, tags: Optional[List[str]] = None, limit: int = 5) -> Optional[Dict[str, Any]]:
        """Searches Gen's private scratchpad."""
        params = {"query": query, "author_ref": author_ref, "limit": limit}
        if tags:
            params['tags'] = ",".join(tags)
        
        return await self._make_request(
            method='GET',
            endpoint='/scratchpad/search',
            params=params
        )

    # --- Main Knowledge Search ---
    async def search_knowledge(self, query: str, author_ref: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Performs a natural language search across the entire knowledge graph."""
        params = {"query": query}
        if author_ref:
            params['author_ref'] = author_ref

        return await self._make_request(
            method='GET',
            endpoint='/search',
            params=params
        )

    # --- Queue Control ---
    async def pause(self) -> Optional[Dict[str, Any]]:
        """Tells TinyGen to pause background processing."""
        return await self._make_request(method='POST', endpoint="/control/pause")

    async def resume(self) -> Optional[Dict[str, Any]]:
        """Tells TinyGen to resume background processing."""
        return await self._make_request(method='POST', endpoint="/control/resume")
        
    async def get_info(self) -> Optional[Dict[str, Any]]:
        """Gets the current status and queue size from TinyGen."""
        return await self._make_request(method='GET', endpoint="/control/info")
        
    async def process_queue(self) -> Optional[Dict[str, Any]]:
        """Tells TinyGen to start processing its queue."""
        return await self._make_request(method='POST', endpoint="/control/process")
