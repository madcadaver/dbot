# web_search.py

import os
import logging
import aiohttp
import asyncio
import json
from bs4 import BeautifulSoup

thought_logger = logging.getLogger('thought')
dev_logger = logging.getLogger('dev')

class WebSearchManager:
    def __init__(self):
        self.searxng_url = os.getenv('SEARXNG_URL', 'http://10.0.1.201:8686')
        self.localai_url = os.getenv('LOCALAI_URL', 'http://10.0.1.101:9090')
        self.localai_2_url = os.getenv('LOCALAI_2_URL') # Secondary URL for offloading
        self.localai_api_key = os.getenv('LOCALAI_API_KEY')
        self.model_name = os.getenv('MODEL_NAME', 'Gen')
        self.timeout_seconds = 180
        self.max_results_for_urls = 10

        self.CHUNK_SIZE = 2000
        self.CHUNK_OVERLAP = 500

    async def get_search_urls(self, query: str) -> list[str]:
        """Performs a web search, reranks results, and returns a list of URL strings."""
        thought_logger.info(f"Getting search URLs for query: '{query}'")
        url = f"{self.searxng_url}/search"
        params = {'q': query, 'format': 'json'}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get('results', [])
                        results_for_rerank = [{'url': r.get('url'), 'content': r.get('content', '')} for r in results if r.get('url')]
                        ranked_results = await self.rerank_results(results_for_rerank, query)
                        ranked_urls = [res['url'] for res in ranked_results[:self.max_results_for_urls]]
                        dev_logger.info(f"Found and reranked {len(ranked_urls)} URLs for query '{query}'.")
                        return ranked_urls
                    else:
                        dev_logger.error(f"SearXNG search failed: {resp.status} - {await resp.text()}")
                        return []
        except Exception as e:
            dev_logger.error(f"Error getting search URLs for '{query}': {e}", exc_info=True)
            return []

    async def extract_facts_from_url(self, url: str, query_context: str) -> list[str]:
        """Fetches a URL, cleans the HTML, chunks the text, and extracts facts using an LLM."""
        thought_logger.info(f"Extracting facts from URL: {url} for query context: '{query_context}'")
        try:
            html_content = await self._fetch_page_content(url)
            if not html_content:
                return []

            soup = BeautifulSoup(html_content, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            raw_text = soup.get_text(separator=' ', strip=True)

            if not raw_text:
                dev_logger.warning(f"No text content found after cleaning HTML for URL: {url}")
                return []

            text_chunks = self._split_text_into_chunks(raw_text)
            dev_logger.debug(f"Split text from {url} into {len(text_chunks)} chunks.")

            all_facts = []
            for chunk in text_chunks:
                facts_from_chunk = await self._call_llm_for_fact_extraction(chunk, query_context)
                all_facts.extend(facts_from_chunk)
                thought_logger.info(f"Extracted {facts_from_chunk}.")

            thought_logger.info(f"Extracted a total of {len(all_facts)} raw facts from {url}.")
            return all_facts

        except Exception as e:
            dev_logger.error(f"Failed to extract facts from URL {url}: {e}", exc_info=True)
            return []

    async def _fetch_page_content(self, url: str) -> str | None:
        """Fetches the HTML content of a single page."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout_seconds, headers={'User-Agent': 'Mozilla/5.0'}) as resp:
                    if resp.status == 200:
                        try:
                            return await resp.text(encoding='utf-8')
                        except UnicodeDecodeError:
                            dev_logger.warning(f"UTF-8 decode failed for {url}, trying latin1")
                            return await resp.text(encoding='latin1')
                    else:
                        dev_logger.error(f"Page fetch failed for {url} with status: {resp.status}")
                        return None
        except asyncio.TimeoutError:
            dev_logger.error(f"Page fetch timed out for {url}")
            return None
        except Exception as e:
            dev_logger.error(f"Generic error fetching page {url}: {e}")
            return None

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """Splits a large text into smaller, overlapping chunks."""
        if not text:
            return []
        return [text[i:i + self.CHUNK_SIZE] for i in range(0, len(text), self.CHUNK_SIZE - self.CHUNK_OVERLAP)]

    async def _call_llm_for_fact_extraction(self, text_chunk: str, query_context: str) -> list[str]:
        """Calls an LLM with a specific prompt to extract facts from a text chunk as a list of strings."""
        system_prompt = (
            "You are a meticulous data extraction engine. Your task is to identify and list all key facts, statements, and data points from the provided text chunk. "
            f"The original user query for context was: '{query_context}'. Focus on information relevant to this query. "
            "Extract each distinct fact or data point as a separate, complete sentence. "
            "Present the output as a JSON array of strings. For example: [\"Fact one.\", \"Fact two.\", \"Data point three.\"] "
            "If the text chunk contains no relevant facts, return an empty JSON array []."
        )
        llm_response = await self._call_localai(system_prompt, text_chunk)

        try:
            clean_response = llm_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            facts = json.loads(clean_response)
            if isinstance(facts, list):
                return facts
            else:
                dev_logger.warning(f"Fact extraction LLM did not return a list. Response: {facts}")
                return []
        except Exception as e:
            dev_logger.error(f"Failed to parse JSON from fact extraction LLM. Response: {llm_response}. Error: {e}")
            return []

    async def _call_rerank_api(self, target_url: str, payload: dict, headers: dict) -> list[int] | None:
        """Helper to make the actual rerank API call to a given URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(target_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ranked_indices = [item['index'] for item in data.get('results', [])]
                        dev_logger.info(f"Reranking successful on {target_url}.")
                        return ranked_indices
                    else:
                        dev_logger.error(f"Reranking API call failed on {target_url} with status {resp.status}: {await resp.text()}")
                        return None
        except Exception as e:
            dev_logger.error(f"Reranking connection error on {target_url}: {e}", exc_info=True)
            return None

    async def rerank_results(self, results: list[dict], query: str) -> list[dict]:
        """Rerank search results using a secondary LocalAI instance with fallback to primary."""
        if not os.getenv('RERANK_MODEL') or not results:
            dev_logger.warning("RERANK_MODEL not set or no results to rank. Using original order.")
            return results
            
        rerank_path = "/v1/rerank"
        headers = {"Content-Type": "application/json"}
        if self.localai_api_key:
            headers["Authorization"] = f"Bearer {self.localai_api_key}"
            
        documents = [result.get('content', '') for result in results]
        payload = {
            "model": os.getenv('RERANK_MODEL', 'jina-reranker-v1-base-en'),
            "query": query,
            "documents": documents
        }

        ranked_indices = None

        # Try secondary URL first if it exists
        if self.localai_2_url:
            dev_logger.debug(f"Attempting rerank on secondary LocalAI: {self.localai_2_url}")
            full_secondary_url = f"{self.localai_2_url.rstrip('/')}{rerank_path}"
            ranked_indices = await self._call_rerank_api(full_secondary_url, payload, headers)

        # If secondary failed or wasn't available, use primary as fallback
        if ranked_indices is None:
            if self.localai_2_url:
                dev_logger.warning("Reranking on secondary instance failed. Falling back to primary.")
            else:
                dev_logger.debug("Secondary instance not configured. Using primary for rerank.")
            
            full_primary_url = f"{self.localai_url.rstrip('/')}{rerank_path}"
            ranked_indices = await self._call_rerank_api(full_primary_url, payload, headers)

        # Process the results if we got any, otherwise return original order
        if ranked_indices is not None:
            ranked_results = [results[i] for i in ranked_indices if i < len(results)]
            return ranked_results
        else:
            dev_logger.error("Reranking failed on all available instances. Returning original order.")
            return results

    async def _call_localai(self, system_prompt: str, user_message: str) -> str:
        """Make a generic call to the primary LocalAI for text generation."""
        url = f"{self.localai_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.localai_api_key:
            headers["Authorization"] = f"Bearer {self.localai_api_key}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        payload = {"model": self.model_name, "messages": messages, "max_tokens": 2048, "temperature": 0.1}
        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with asyncio.timeout(self.timeout_seconds):
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                response = data["choices"][0]["message"]["content"].strip()
                                return response
                            else:
                                dev_logger.error(f"LocalAI request failed (attempt {attempt + 1}/{retries}): {resp.status} - {await resp.text()}")
            except asyncio.TimeoutError:
                dev_logger.error(f"LocalAI request timed out (attempt {attempt + 1}/{retries})")
            except Exception as e:
                dev_logger.error(f"LocalAI request error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
        return "[]"
