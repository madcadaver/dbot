# media_manager.py

import os
import logging
import aiohttp
import base64
import io
import asyncio
from io import BytesIO
import discord
from PIL import Image
from pathlib import Path
import json

thought_logger = logging.getLogger('thought')
dev_logger = logging.getLogger('dev')

class MediaManager:
    def __init__(self):
        self.localai_url = os.getenv('LOCALAI_URL', 'http://10.0.1.101:9090')
        self.localai_api_key = os.getenv('LOCALAI_API_KEY')
        self.comfy_url = os.getenv('COMFY_URL')
        self.sd_model = os.getenv('SD_MODELL', 'stablediffusion')
        self.vl_model = os.getenv('VL_MODEL', 'moondream2')
        self.image_size = os.getenv('IMAGE_SIZE', '768x768')
        self.max_image_size = int(os.getenv('MAX_IMAGE_SIZE', '1280'))
        self.image_n = int(os.getenv('IMAGE_N', '2'))
        self.image_steps = int(os.getenv('IMAGE_STEPS', '40'))
        self.image_cfg_scale = float(os.getenv('IMAGE_CFG_SCALE', '6.5'))
        self.timeout_seconds = 60
        dev_logger.debug(f"Loaded IMAGE_SIZE={self.image_size}, IMAGE_N={self.image_n} from .env")

    async def generate_image_from_comfyui(self, prompt, image_n=None, image_size=None):
        """Generate images using ComfyUI and return Discord-ready files."""
        image_n = image_n if image_n is not None else self.image_n
        image_size = image_size if image_size is not None else self.image_size
        headers = {"Content-Type": "application/json"}
        if self.localai_api_key:
            headers["Authorization"] = f"Bearer {self.localai_api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "nomodel",
                    "messages": [{"role": "user", "content": "Say this is a test!"}]
                }
                dev_logger.debug("Attempting to free VRAM by calling LocalAI with 'nomodel'")
                async with session.post(self.localai_url + '/v1/chat/completions', json=payload, headers=headers) as resp:
                    dev_logger.debug(f"LocalAI nomodel call response status: {resp.status}")
                    thought_logger.info("Waiting 10 seconds for VRAM to free after nomodel call...")
                    await asyncio.sleep(10)
        except Exception as e:
            dev_logger.debug(f"Non-critical error during nomodel call: {e}")
            pass

        # Determine which workflow to use based on the prompt content.
        workflow_filename = os.environ.get('WFLOW') # Default workflow
        if 'anime' in prompt.lower():
            dev_logger.info("'anime' keyword detected in prompt, checking for alternate workflow.")
            anime_workflow = os.environ.get('WFLOW_IL')
            if anime_workflow:
                workflow_filename = anime_workflow
                dev_logger.info(f"Using anime-specific workflow: {workflow_filename}")
            else:
                dev_logger.info("WFLOW_IL not defined, falling back to default workflow.")

        if not workflow_filename:
            dev_logger.error("No workflow filename could be determined. WFLOW environment variable is not set.")
            return []

        try:
            workflow_path = Path('data/workflows') / workflow_filename
            # CORRECTED: Reverted to the original config file naming convention.
            config_filename = workflow_filename.replace('.json', '-conf.json')
            config_path = Path('data/config') / config_filename
            
            dev_logger.info(f"Loading workflow: {workflow_path}")
            dev_logger.info(f"Loading config: {config_path}")

            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError as e:
            dev_logger.error(f"Could not find workflow or config file: {e}")
            return []
        except Exception as e:
            dev_logger.error(f"Failed to load workflow or config file: {e}")
            return []

        width, height = map(int, image_size.split('x'))
        if width > self.max_image_size or height > self.max_image_size:
            width = min(width, self.max_image_size)
            height = min(height, self.max_image_size)
            
        # Update workflow with prompt, size, and batch details from config
        try:
            workflow[config['prompt_node']]['inputs'][config['prompt_key']] = prompt
            workflow[config['size_node']]['inputs'][config['width_key']] = width
            workflow[config['size_node']]['inputs'][config['height_key']] = height
            workflow[config['size_node']]['inputs'][config['batch_size_key']] = image_n
        except KeyError as e:
            dev_logger.error(f"KeyError when configuring workflow. Check your config file '{config_filename}'. Missing key: {e}")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                prompt_id = None
                retries = 3
                headers = {"Content-Type": "application/json"}
                for attempt in range(retries):
                    try:
                        async with session.post(self.comfy_url + '/prompt', json={"prompt": workflow}, headers=headers) as resp:
                            if resp.status != 200:
                                response_text = await resp.text()
                                dev_logger.error(f"ComfyUI prompt queuing failed (attempt {attempt + 1}/{retries}): {resp.status} - {response_text}")
                                if attempt == retries - 1:
                                    return []
                                await asyncio.sleep(2)
                                continue
                            data = await resp.json()
                            prompt_id = data['prompt_id']
                            thought_logger.info(f"ComfyUI prompt queued with ID: {prompt_id}")
                            break
                    except Exception as e:
                        dev_logger.error(f"ComfyUI prompt queuing error (attempt {attempt + 1}/{retries}): {e}")
                        if attempt == retries - 1:
                            return []
                        await asyncio.sleep(2)
                if not prompt_id:
                    dev_logger.error("Failed to queue prompt after all retries")
                    return []

                while True:
                    async with session.get(self.comfy_url + '/history/' + prompt_id) as resp:
                        if resp.status != 200:
                            dev_logger.debug(f"Polling attempt failed for prompt ID {prompt_id}: {resp.status}")
                            await asyncio.sleep(1)
                            continue
                        history = await resp.json()
                        if prompt_id in history:
                            # Check for completion status
                            if history[prompt_id].get('status', {}).get('completed') and 'outputs' in history[prompt_id]:
                                break # Success
                            # Check for errors during execution
                            if not history[prompt_id].get('status', {}).get('completed'):
                                if 'messages' in history[prompt_id].get('status', {}):
                                    for msg in history[prompt_id]['status']['messages']:
                                        if msg[0] == 'execution_error':
                                            dev_logger.error(f"ComfyUI generation failed: {msg[1].get('exception_message', 'No details')}")
                                            return []
                            await asyncio.sleep(1)
                            continue
                    await asyncio.sleep(1)

                files = []
                for node_id, node_output in history[prompt_id]['outputs'].items():
                    if 'images' in node_output:
                        for idx, image in enumerate(node_output['images']):
                            filename = image['filename']
                            subfolder = image.get('subfolder', '')
                            params = {'filename': filename, 'type': 'output', 'subfolder': subfolder}
                            async with session.get(self.comfy_url + '/view', params=params) as img_resp:
                                if img_resp.status != 200:
                                    dev_logger.error(f"Failed to download image {filename}: {img_resp.status}")
                                    continue
                                img_data = await img_resp.read()
                                file = discord.File(fp=BytesIO(img_data), filename=f"gen_image_{idx}.png")
                                files.append(file)
                thought_logger.info(f"Generated {len(files)} images via ComfyUI")
                return files
        except Exception as e:
            dev_logger.error(f"ComfyUI image generation error: {e}", exc_info=True)
            return []

    async def generate_image_from_localai(self, prompt, neg_prompt=None, image_n=None, image_size=None):
        """Generate images using Stable Diffusion or ComfyUI and return Discord-ready files."""
        image_n = image_n if image_n is not None else self.image_n
        image_size = image_size if image_size is not None else self.image_size
        if self.comfy_url:
            return await self.generate_image_from_comfyui(prompt, image_n, image_size)

        url = f"{self.localai_url}/v1/images/generations"
        headers = {"Content-Type": "application/json"}
        if self.localai_api_key:
            headers["Authorization"] = f"Bearer {self.localai_api_key}"
        width, height = map(int, image_size.split('x'))
        if width > self.max_image_size or height > self.max_image_size:
            width = min(width, self.max_image_size)
            height = min(height, self.max_image_size)
        payload = {
            "model": self.sd_model,
            "prompt": prompt,
            "negative_prompt": neg_prompt if neg_prompt else "",
            "n": image_n,
            "size": f"{width}x{height}",
            "num_inference_steps": self.image_steps,
            "guidance_scale": self.image_cfg_scale
        }
        thought_logger.info(f"Gen is generating an image with prompt: '{prompt}'")
        dev_logger.debug(f"Image generation payload: {payload}")
        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with asyncio.timeout(self.timeout_seconds):
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                images = []
                                async with aiohttp.ClientSession() as download_session:
                                    for img_data in data.get("data", []):
                                        if "url" in img_data:
                                            image_url = img_data["url"]
                                            async with download_session.get(image_url, headers=headers) as img_resp:
                                                if img_resp.status == 200:
                                                    img_bytes = await img_resp.read()
                                                    images.append(discord.File(BytesIO(img_bytes), f"gen_image_{len(images)}.png"))
                                                else:
                                                    dev_logger.error(f"Image download failed from {image_url}: {img_resp.status}")
                                        elif "b64_json" in img_data:
                                            img_bytes = base64.b64decode(img_data["b64_json"])
                                            images.append(discord.File(BytesIO(img_bytes), f"gen_image_{len(images)}.png"))
                                thought_logger.info(f"Generated {len(images)} images")
                                return images
                            else:
                                dev_logger.error(f"Image generation failed (attempt {attempt + 1}/{retries}): {resp.status}")
            except asyncio.TimeoutError:
                dev_logger.error(f"Image generation timed out (attempt {attempt + 1}/{retries})")
            except Exception as e:
                dev_logger.error(f"Image generation error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
        return []

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """Analyze an image using the vision model and return a description."""
        if not self.vl_model:
            thought_logger.warning("VL_MODEL is not set. Vision functionality disabled.")
            return ""
        try:
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            if max(width, height) > self.max_image_size:
                if width > height:
                    new_width = self.max_image_size
                    new_height = int(height * (self.max_image_size / width))
                else:
                    new_height = self.max_image_size
                    new_width = int(width * (self.max_image_size / height))
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            format = image.format or 'JPEG'
            with io.BytesIO() as output:
                image.save(output, format=format)
                resized_image_data = output.getvalue()
            base64_image = base64.b64encode(resized_image_data).decode('utf-8')
            mime_type = 'jpeg' if format == 'JPEG' else 'png' if format == 'PNG' else 'gif' if format == 'GIF' else 'jpeg'

            messages = [
                {"role": "system", "content": "You are a helpful assistant with vision capabilities."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{base64_image}"}}
                    ]
                }
            ]
            url = f"{self.localai_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.localai_api_key:
                headers["Authorization"] = f"Bearer {self.localai_api_key}"
            data = {"model": self.vl_model, "messages": messages, "max_tokens": 100}
            thought_logger.info("Gen is analyzing an image")
            dev_logger.debug(f"Analyzing image.")
            retries = 3
            for attempt in range(retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with asyncio.timeout(self.timeout_seconds):
                            async with session.post(url, headers=headers, json=data) as resp:
                                if resp.status == 200:
                                    result = await resp.json()
                                    description = result['choices'][0]['message']['content']
                                    thought_logger.info(f"Image analyzed: '{description}'")
                                    dev_logger.debug(f"Image analysis result: {description}")
                                    return description
                                else:
                                    dev_logger.error(f"Image analysis failed (attempt {attempt + 1}/{retries}): {resp.status}")
                except asyncio.TimeoutError:
                    dev_logger.error(f"Image analysis timed out (attempt {attempt + 1}/{retries})")
                except Exception as e:
                    dev_logger.error(f"Image analysis error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
            return ""
        except Exception as e:
            dev_logger.error(f"Image analysis error: {e}")
            return ""
