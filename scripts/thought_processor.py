# thought_processor.py

import os
import logging
import json
import asyncio
import aiohttp
import re
from pathlib import Path

from media_manager import MediaManager
from web_search import WebSearchManager
from user_profiles import UserProfileManager
import capabilities

from tinygen_controller import TinyGenController

import sentencepiece as spm

thought_logger = logging.getLogger('thought')
dev_logger = logging.getLogger('dev')

FALLBACK_PROMPT_OVERHEAD_TOKENS = 2000

class ThoughtProcessor:
    def __init__(self,
                 media_manager: MediaManager,
                 web_search_manager: WebSearchManager,
                 user_profile_manager: UserProfileManager):

        self.tinygen_controller = TinyGenController()

        self.user_profile_manager = user_profile_manager
        self.media_manager = media_manager
        self.web_search_manager = web_search_manager

        self.localai_url = os.getenv('LOCALAI_URL', 'http://localhost:8080')
        self.localai_api_key = os.getenv('LOCALAI_API_KEY')
        self.model_name = os.getenv('MODEL_NAME', 'gpt-4')
        self.timeout_seconds = int(os.getenv('LOCALAI_TIMEOUT_SECONDS', 120))
        self.llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        
        self.gen_name = "Gen"
        self.main_system_prompt_base = ""
        self.bot_user_id = None

        self.tool_schemas = capabilities.get_tool_schemas()
        dev_logger.debug(f"ThoughtProcessor initialized with {len(self.tool_schemas)} tool schemas from capabilities.")

        self.tokenizer = None
        try:
            tokenizer_model_path = Path(os.getenv('TOKENIZER_MODEL_PATH', 'models/tokenizer.model'))
            if tokenizer_model_path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.load(str(tokenizer_model_path))
            else:
                dev_logger.warning(f"Tokenizer model not found at '{tokenizer_model_path}'.")
        except Exception as e:
            dev_logger.error(f"Failed to load SentencePiece tokenizer: {e}.")
            self.tokenizer = None

    def set_bot_user_id(self, bot_user_id: str):
        self.bot_user_id = str(bot_user_id)
        dev_logger.debug(f"ThoughtProcessor: bot_user_id set to {self.bot_user_id}")

    def set_gen_profile(self, name: str, personality: str, appearance: str, birthdate: str, main_system_prompt: str):
        self.gen_name = name
        self.main_system_prompt_base = main_system_prompt 
        dev_logger.info(f"ThoughtProcessor Gen profile updated: Name={name}.")

    def get_prompt_overhead_tokens(self, message_details_from_cm: str) -> int:
        if not self.tokenizer: return FALLBACK_PROMPT_OVERHEAD_TOKENS
        system_prompt_text = (f"{self.main_system_prompt_base} {message_details_from_cm} "
                              "Based on our conversation history, decide the best way to proceed.")
        try:
            system_tokens = len(self.tokenizer.encode(system_prompt_text))
            tools_tokens = len(self.tokenizer.encode(json.dumps(self.tool_schemas)))
            return system_tokens + tools_tokens + 50
        except Exception:
            return FALLBACK_PROMPT_OVERHEAD_TOKENS

    async def _call_localai_llm(self, messages: list, tools=None, tool_choice="auto", temperature=None, max_tokens: int = 2048):
        if not self.localai_url: return {"error": "LocalAI URL not configured."}
        endpoint, headers = f"{self.localai_url}/v1/chat/completions", {"Content-Type": "application/json"}
        if self.localai_api_key: headers["Authorization"] = f"Bearer {self.localai_api_key}"
        payload = {"model": self.model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature or self.llm_temperature}
        if tools: payload.update({"tools": tools, "tool_choice": tool_choice})
        
        dev_logger.debug(f"Calling LocalAI LLM: Endpoint={endpoint}, Model={self.model_name}, Temp={payload['temperature']}, Max_Tokens={payload['max_tokens']}")
        
        if dev_logger.level == logging.DEBUG:
            dev_logger.debug(f"Full payload to be sent to LocalAI LLM: {json.dumps(payload, indent=2)}")

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session, session.post(endpoint, json=payload, headers=headers, timeout=self.timeout_seconds) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        dev_logger.debug(f"LocalAI raw response (status 200): {json.dumps(data, indent=2)}")
                        return data
                    dev_logger.error(f"LocalAI API call failed (attempt {attempt+1}): {resp.status}, {await resp.text()}")
            except Exception as e:
                dev_logger.error(f"LocalAI call error (attempt {attempt+1}): {e}", exc_info=True)
            if attempt < 2: await asyncio.sleep(1)
        return {"error": "LocalAI call failed after all retries."}

    async def get_next_decision(self, base_history: list, current_interaction_turns: list, user_message: dict, long_term_history_str: str, message_queue_str: str, persona_data: dict, max_tokens: int):
        messages = []
        
        # 1. Long-Term Memory
        if long_term_history_str:
            messages.append({"role": "system", "content": long_term_history_str})

        # 2. Short-Term History
        messages.extend(base_history)

        # 3. NEW: Real-time Message Queue
        if message_queue_str:
            messages.append({"role": "system", "content": message_queue_str})

        # 4. System Prompt
        system_prompt = f"{self.main_system_prompt_base} {persona_data.get('details_for_prompt', '')}"
        
        if current_interaction_turns:
            system_prompt += "\n\nCRITICAL INSTRUCTION: Analyze the results of the tool(s) you just used. Your previous turn resulted in new information. You must now decide whether to use another tool or to respond to the user with the information you have gathered."
        messages.append({"role": "system", "content": system_prompt})

        # 5. Current Interaction
        messages.append(user_message)
        messages.extend(current_interaction_turns)

        # This part remains unchanged
        response = await self._call_localai_llm(messages, tools=self.tool_schemas, max_tokens=max_tokens)
        
        if "error" in response: return {"type": "error", "content": f"Neko's brain fuzzy... {response['error']}"}
        try:
            message = response["choices"][0]["message"]
            if message.get("tool_calls"):
                call = message["tool_calls"][0]
                return {"type": "tool_call", "name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"]), "id": call["id"]}
            if message.get("content"):
                return {"type": "tool_call", "name": "respond_to_user", "arguments": {"text_to_send": message["content"], "response_type_guidance": "default"}, "id": "direct_resp_" + os.urandom(4).hex()}
            return {"type": "error", "content": "I'm speechless, neko!"}
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            return {"type": "error", "content": f"A glitch in my circuits, neko! {e}"}


    async def _filter_data_for_relevance(self, query: str, facts: list[str]) -> (str, list[str]):
        if not facts: return "No facts were extracted from the source.", []
        
        facts_str = "\n".join(f'- {f}' for f in facts)
        
        prompt = (f"""You are a filtering agent. The user's original query was: '{query}'.
Review the following list of facts.
1. Create a concise, coherent paragraph summarizing only the most relevant facts.
2. List the specific facts you used for the summary.

Respond with a single JSON object with two keys: "summary" (a string) and "relevant_facts" (a list of strings).
Facts:
{facts_str}""")

        response = await self._call_localai_llm([{"role": "system", "content": prompt}, {"role": "user", "content": "Please provide the JSON response."}], temperature=0.2, max_tokens=1024)
        
        try:
            content = response["choices"][0]["message"]["content"].strip()
            json_match = re.search(r'```json\n({.*?})\n```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content)

            summary = data.get("summary", "Could not produce a summary.")
            relevant_facts = data.get("relevant_facts", [])
            return summary, relevant_facts
        except (Exception) as e:
            dev_logger.error(f"Error while filtering facts for relevance: {e}")
            return f"Error while filtering facts: {response.get('error', 'Could not parse LLM response.')}", facts[:3]

    async def execute_tool(self, tool_name: str, tool_args: dict, tool_call_id: str, interaction_state: dict, author_id: str, author_username: str):
        thought_logger.info(f"Executing tool: '{tool_name}' with args: {tool_args} for user '{author_username}'")
        try:
            if tool_name == "store_knowledge":
                unstructured_text = tool_args.get("unstructured_text")
                if not unstructured_text:
                    return {"type": "error", "content": "LLM didn't provide text to store!"}

                text_for_api = self._replace_aliases_with_user_id_format(unstructured_text, self.user_profile_manager)
                author_reference = f"User (user_id: {self.bot_user_id})"

                api_result = await self.tinygen_controller.store_knowledge(
                    unstructured_text=text_for_api,
                    author_ref=author_reference
                )
                
                history_result = self._replace_user_id_format_with_aliases(api_result, self.user_profile_manager)
                
                return {"type": "intermediate_tool_result", "tool_call_id": tool_call_id, "tool_name": tool_name, "content": history_result}

            elif tool_name == "respond_to_user":
                return {"type": "terminal_result", "content": tool_args.get("text_to_send", "I'm speechless, neko!")}

            elif tool_name == "generate_image":
                prompt = tool_args.get("image_generation_prompt")
                comment = tool_args.get("comment_for_image", "Here you go, darling!")
                if not prompt: 
                    return {"type": "error", "content": "The LLM forgot what to draw!"}
                
                files = await self.media_manager.generate_image_from_localai(prompt)

                result_package = {
                    "type": "terminal_result",
                    "content": (comment, files),
                    "prompt_for_logging": prompt # Add this key for accurate logging
                }
                
                if not files:
                    result_package.update({"type": "error", "content": "My art tools are jammed!"})

                return result_package

            elif tool_name == "perform_web_search":
                query = tool_args.get("search_query_for_web")
                if not query: return {"type": "intermediate_tool_result", "tool_call_id": tool_call_id, "tool_name": tool_name, "content": "[No search results found.]"}
                urls = await self.web_search_manager.get_search_urls(query)
                if not urls: return {"type": "intermediate_tool_result", "tool_call_id": tool_call_id, "tool_name": tool_name, "content": "[No search results found.]"}
                
                facts = await self.web_search_manager.extract_facts_from_url(urls[0], query)
                summary, relevant_facts = await self._filter_data_for_relevance(interaction_state.get('original_user_message', query), facts)
                
                if relevant_facts:
                    text_to_store = "\n".join(relevant_facts)
                    text_for_api = self._replace_aliases_with_user_id_format(text_to_store, self.user_profile_manager)
                    author_reference = f"User (user_id: {self.bot_user_id})"
                    storage_result = await self.tinygen_controller.store_knowledge(
                        unstructured_text=text_for_api,
                        author_ref=author_reference
                    )
                    dev_logger.info(f"Relevant web search facts stored. Response: {storage_result}")

                # The 'content' of the web search result should be the summary for the history
                return {"type": "intermediate_tool_result", "tool_call_id": tool_call_id, "tool_name": tool_name, "content": summary}

            elif tool_name == "overthink_input":
                thought_process = tool_args.get("detailed_thought_process", "I've analyzed the situation...")
                return {"type": "intermediate_tool_result", "tool_call_id": tool_call_id, "tool_name": tool_name, "content": thought_process}

            elif tool_name == "inquire_for_details":
                question = tool_args.get("clarifying_question_to_ask")
                if not question: return {"type": "error", "content": "The LLM wanted to ask something but forgot what!"}
                return {"type": "terminal_result", "content": question}

            else:
                dev_logger.warning(f"Attempted to execute unknown tool: {tool_name}")
                return {"type": "error", "content": f"I don't know the tool '{tool_name}', neko!"}
        
        except Exception as e:
            dev_logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return {"type": "error", "content": f"Something went wrong with my {tool_name} tool, neko! ({str(e)})"}

    def _replace_aliases_with_user_id_format(self, text: str, upm: UserProfileManager) -> str:
        if not upm: return text
        profiles = upm.get_all_user_profiles_for_mention_mapping()
        modified_text = text
        for name, user_id in profiles:
            if name and user_id:
                pattern = r'\b' + re.escape(name) + r'\b'
                replacement_format = f"User (user_id: {user_id})"
                modified_text = re.sub(pattern, replacement_format, modified_text, flags=re.IGNORECASE)
        return modified_text

    def _replace_user_id_format_with_aliases(self, text: str, upm: UserProfileManager) -> str:
        if not upm: return text
        return re.sub(r"User \(user_id: (\d+)\)", lambda m: upm.get_user_alias(m.group(1)) or m.group(0), text)

    def replace_mentions_with_aliases(self, text: str, upm: UserProfileManager) -> str:
        if not upm: return text
        def repl(m):
            uid = m.group(1)
            if self.bot_user_id and uid == self.bot_user_id: return self.gen_name
            return upm.get_user_alias(uid) or m.group(0)
        return re.sub(r'<@!?(\d+)>', repl, text)

    def replace_aliases_with_mentions(self, text: str, upm: UserProfileManager) -> str:
        if not upm: return text
        profiles = upm.get_all_user_profiles_for_mention_mapping()
        for name, user_id in profiles:
            if name and user_id:
                pattern = r'\b' + re.escape(name) + r'\b'
                text = re.sub(pattern, f"<@{user_id}>", text, flags=re.IGNORECASE)
        return text
