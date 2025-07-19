# embeddings.py

import os
import logging
import aiohttp
import asyncio
import numpy as np

dev_logger = logging.getLogger('dev')

async def generate_embedding(text, use_secondary=True):
    """
    Generate an embedding for the given text using LocalAI.

    Args:
        text (str): The text to embed.
        use_secondary (bool): Use LOCALAI_2_URL if True, else LOCALAI_URL.

    Returns:
        np.ndarray: A 384-dimensional embedding vector, or zeros on failure.
    """
    base_url = os.getenv('LOCALAI_2_URL', os.getenv('LOCALAI_URL', 'http://10.0.1.101:9090')) if use_secondary else os.getenv('LOCALAI_URL', 'http://10.0.1.101:9090')
    url = f"{base_url}/v1/embeddings"
    api_key = os.getenv('LOCALAI_2_API_KEY', os.getenv('LOCALAI_API_KEY')) if use_secondary else os.getenv('LOCALAI_API_KEY')
    model = os.getenv('EMBEDDINGS_2_MODEL', os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')) if use_secondary else os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')
    dev_logger.debug(f"Generating embedding for text: '{text[:50]}...' using URL: {url}, model: {model}")
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": text if text.strip() else "default"}
    retries = 3
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with asyncio.timeout(30):
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if "data" not in data or not data["data"]:
                                dev_logger.error(f"No 'data' in embedding response: {data}")
                                raise ValueError("No 'data' in embedding response")
                            embedding = np.array(data["data"][0]["embedding"])
                            dev_logger.debug(f"Generated embedding with shape: {embedding.shape}")
                            return embedding
                        dev_logger.error(f"Embedding generation failed (attempt {attempt + 1}/{retries}): {resp.status} - {await resp.text()}")
            except asyncio.TimeoutError:
                dev_logger.error(f"Embedding generation timed out (attempt {attempt + 1}/{retries})")
            except Exception as e:
                dev_logger.error(f"Embedding generation error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
            if use_secondary and attempt == retries - 1:
                dev_logger.warning("Secondary LocalAI failed, falling back to primary LocalAI")
                return await generate_embedding(text, use_secondary=False)
        dev_logger.error("Failed to generate embedding after all retries")
        return np.zeros(384)
