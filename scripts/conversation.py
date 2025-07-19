# conversation.py

import os
import logging
import discord
import re
import json
from pathlib import Path
from datetime import datetime, timezone as dt_timezone, timedelta
import pytz
import asyncio

from thought_processor import ThoughtProcessor
from user_profiles import UserProfileManager
from knowledge_graph import Neo4jManager
from media_manager import MediaManager
from web_search import WebSearchManager
from database_manager import DatabaseManager
from tinygen_controller import TinyGenController
from history_manager import HistoryManager

import sentencepiece as spm

thought_logger = logging.getLogger('thought')
dev_logger = logging.getLogger('dev')
neo4j_logger = logging.getLogger('neo4j')

class ConversationManager:
    def __init__(self,
                 user_profile_manager: UserProfileManager,
                 neo4j_manager: Neo4jManager,
                 media_manager: MediaManager,
                 web_search_manager: WebSearchManager,
                 database_manager: DatabaseManager,
                 tinygen_controller: TinyGenController,
                 message_queue: asyncio.Queue): # Modified

        self.user_profile_manager = user_profile_manager
        self.neo4j_manager = neo4j_manager
        self.media_manager = media_manager
        self.web_search_manager = web_search_manager
        self.database_manager = database_manager
        self.tinygen_controller = tinygen_controller
        self.message_queue = message_queue # New

        self.thought_processor = ThoughtProcessor(
            media_manager=self.media_manager,
            web_search_manager=self.web_search_manager,
            user_profile_manager=self.user_profile_manager
        )

        self.gen_profile_path = Path("data/gen_profile.json")
        self.gen_profile = self._load_gen_profile()
        self.timezone = pytz.timezone(os.getenv('TIMEZONE', 'UTC'))
        self.gen_name = self.gen_profile.get('name', 'Gen')

        self.history_primary_fetch_limit = int(os.getenv('HISTORY_PRIMARY_FETCH_LIMIT', "30"))
        self.history_max_turns = int(os.getenv('HISTORY_MAX_TURNS', "20"))
        self.llm_max_context_tokens = int(os.getenv('LLM_MAX_CONTEXT_TOKENS', '8192'))
        self.minimum_budget_for_history_tokens = int(os.getenv('MINIMUM_BUDGET_FOR_HISTORY_TOKENS', '500'))
        self.llm_max_output_tokens = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '2048'))
        self.MAX_TOOL_ITERATIONS = 12

        self.bot_user_id = None
        self.known_public_channel_ids: list[str] = []
        self.channel_name_map: dict[str, str] = {}

        self._initialize_gen_profile()
        self._initialize_tokenizer()

        self.history_manager = HistoryManager(
            neo4j_manager=self.neo4j_manager,
            user_profile_manager=self.user_profile_manager,
            thought_processor=self.thought_processor,
            tokenizer=self.tokenizer,
            channel_name_map=self.channel_name_map,
            bot_user_id=self.bot_user_id
        )

        dev_logger.debug("ConversationManager initialized.")

    def _initialize_gen_profile(self):
        date_format = "%A, %B %d, %Y, %H:%M %Z"
        current_time_str = datetime.now(self.timezone).strftime(date_format)
        main_system_prompt_for_tp = (
            f"You are {self.gen_name}, a {self.gen_profile.get('personality', 'fiery, playful, and moody')}. "
            f"Your appearance is: {self.gen_profile.get('appearance', 'steampunk style with red hair')}. You were born on {self.gen_profile.get('birthdate', 'March 15, 1992')}. "
            f"Its {current_time_str}. "
        )
        self.thought_processor.set_gen_profile(
            name=self.gen_name,
            personality=self.gen_profile.get('personality'),
            appearance=self.gen_profile.get('appearance'),
            birthdate=self.gen_profile.get('birthdate'),
            main_system_prompt=main_system_prompt_for_tp
        )

    def _initialize_tokenizer(self):
        self.tokenizer = None
        try:
            tokenizer_model_path = Path(os.getenv('TOKENIZER_MODEL_PATH', 'models/tokenizer.model'))
            if tokenizer_model_path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.load(str(tokenizer_model_path))
                dev_logger.info(f"ConversationManager: SentencePiece tokenizer loaded from {tokenizer_model_path}.")
            else:
                dev_logger.warning(f"ConversationManager: Tokenizer model not found at '{tokenizer_model_path}'. Token counts will be estimated.")
        except Exception as e:
            dev_logger.error(f"ConversationManager: Failed to load SentencePiece tokenizer: {e}. Token counts will be estimated.", exc_info=True)

    def set_bot_user_id(self, bot_user_id: str):
        self.bot_user_id = str(bot_user_id)
        self.thought_processor.set_bot_user_id(self.bot_user_id)
        dev_logger.info(f"ConversationManager: bot_user_id set to {self.bot_user_id}")

    def set_channel_name_map(self, id_to_name_map: dict[str, str]):
        self.channel_name_map = id_to_name_map
        self.known_public_channel_ids = list(id_to_name_map.keys())
        dev_logger.info(f"ConversationManager: Channel name map set. Count: {len(self.channel_name_map)}")

    def _load_gen_profile(self):
        try:
            if self.gen_profile_path.exists():
                with open(self.gen_profile_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            dev_logger.error(f"Error loading gen_profile.json: {e}", exc_info=True)
            return {}

    async def _store_message_in_neo4j(self, discord_message_object: discord.Message, role: str,
                                      interaction_id_for_message: str, content_to_store: str = None):
        if not self.neo4j_manager: return
        try:
            content = content_to_store if content_to_store is not None else discord_message_object.content
            token_count = len(self.tokenizer.encode(content)) if self.tokenizer else len(content) // 4
            await self.neo4j_manager.create_message_node(
                message_id=str(discord_message_object.id), author_user_id=str(discord_message_object.author.id),
                interaction_id=interaction_id_for_message, channel_id=str(discord_message_object.channel.id),
                is_dm=isinstance(discord_message_object.channel, discord.DMChannel), role=role, content_to_store=content,
                timestamp=int(discord_message_object.created_at.replace(tzinfo=dt_timezone.utc).timestamp()), token_count=token_count,
                length_chars=len(content), has_attachments=bool(discord_message_object.attachments)
            )
        except Exception as e:
            dev_logger.error(f"Failed to store {role} message ID {discord_message_object.id} in Neo4j: {e}", exc_info=True)

    async def _handle_attachments(self, message: discord.Message, initial_content: str, interaction_id: str, interaction_timestamp: int) -> str:
        appended_texts = []
        if not message.attachments:
            return initial_content
        for attachment in message.attachments:
            attachment_summary_prefix = f"[Attachment: {attachment.filename} - Type: {attachment.content_type} - "
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    image_data = await attachment.read()
                    vision_prompt = f"Describe this image. The user who sent it (named '{message.author.display_name}') also wrote: '{message.content}'"
                    description = await self.media_manager.analyze_image(image_data=image_data, prompt=vision_prompt)
                    if description and description.strip():
                        appended_texts.append(f"{attachment_summary_prefix}Summary: {description.strip()}]")
                        await self.neo4j_manager.insert_action(interaction_id, str(message.channel.id), "AnalyzeImageAttachment", interaction_timestamp, f"User sent image: {attachment.filename}", result_summary=description.strip()[:250])
                    else:
                        appended_texts.append(f"{attachment_summary_prefix}Could not get a description.]")
                except Exception as e_vision:
                    dev_logger.error(f"Error analyzing image attachment '{attachment.filename}': {e_vision}", exc_info=True)
                    appended_texts.append(f"{attachment_summary_prefix}Error during analysis.]")
            else:
                appended_texts.append(f"{attachment_summary_prefix}Processing not yet implemented for this type.]")
        if appended_texts:
            separator = "\n" if initial_content.strip() else ""
            return f"{initial_content}{separator}" + "\n".join(appended_texts).strip()
        return initial_content

    async def _handle_alias_change(self, message: discord.Message, user_id_str: str, username_str: str, author_display_name: str, interaction_id: str, interaction_timestamp: int) -> bool:
        new_alias_from_request = self.is_alias_change_request(message.content)
        if new_alias_from_request:
            self.user_profile_manager.update_user_alias(user_id_str, new_alias_from_request, username_str)
            await self.neo4j_manager.insert_action(interaction_id, str(message.channel.id), "UpdateAlias", interaction_timestamp, f"User requested alias change to '{new_alias_from_request}'.", f"Alias set to {new_alias_from_request}")
            confirmation_text = f"Alright, {author_display_name}! I'll call you {new_alias_from_request} from now on, neko!"
            sent_msg = await message.channel.send(confirmation_text)
            await self._store_message_in_neo4j(sent_msg, "assistant", interaction_id, confirmation_text)
            return True
        return False

    def is_alias_change_request(self, message_text: str) -> str | None:
        patterns = [r"(?:call\s+me|my\s+name\s+is|i\s+want\s+to\s+be\s+called)\s+([\w\s'-]+)"]
        for pattern in patterns:
            match = re.search(pattern, message_text, re.IGNORECASE)
            if match:
                new_alias = match.group(1).strip()
                if 0 < len(new_alias) < 50 and new_alias.lower() != self.gen_name.lower():
                    return new_alias
        return None

    def _get_channel_ids_for_priority_fetch(self, current_message: discord.Message) -> set[str]:
        priority_channel_ids = {str(current_message.channel.id)}
        user_profile = self.neo4j_manager.get_user(str(current_message.author.id))
        if user_profile:
            if user_profile.get('dm_channel_id'): priority_channel_ids.add(user_profile['dm_channel_id'])
            if user_profile.get('last_active_channel_id'): priority_channel_ids.add(user_profile['last_active_channel_id'])
        if self.bot_user_id:
            bot_profile = self.neo4j_manager.get_user(self.bot_user_id)
            if bot_profile and bot_profile.get('last_active_channel_id'):
                priority_channel_ids.add(bot_profile['last_active_channel_id'])
        return priority_channel_ids

    async def store_message_for_context(self, message: discord.Message, existing_interaction_id: str = None):
        dev_logger.debug(f"Storing message ID {message.id} for context.")
        user_id_str, username_str = str(message.author.id), message.author.name
        self.user_profile_manager.add_new_user(user_id=user_id_str, username=username_str)
        interaction_timestamp = int(message.created_at.replace(tzinfo=dt_timezone.utc).timestamp())
        interaction_id = existing_interaction_id or str(message.id)
        if not existing_interaction_id:
            await self.neo4j_manager.create_interaction(user_id_str, interaction_id, interaction_timestamp)

        role = "assistant" if str(message.author.id) == self.bot_user_id else "user"
        content_to_store = message.content
        if role == "user":
            content_to_store = await self._handle_attachments(message, message.content, interaction_id, interaction_timestamp)

        await self._store_message_in_neo4j(message, role, interaction_id, content_to_store)
        dev_logger.info(f"Successfully stored message ID {message.id} from '{username_str}'.")

    def _format_message_queue_for_prompt(self, current_channel_id: str) -> str:
        """
        New: Peeks at the message queue and formats it for the LLM prompt.
        Filters to show only messages from the current channel.
        """
        MAX_QUEUE_DISPLAY = 3
        
        # Access the underlying deque to peek without removing items
        queued_messages = list(self.message_queue._queue)
        
        if not queued_messages:
            return ""

        # Filter messages for the current channel
        relevant_messages = [msg for msg in queued_messages if str(msg.channel.id) == current_channel_id]
        if not relevant_messages:
            return ""

        formatted_queue = ["[MESSAGES AWAITING YOUR ATTENTION IN THIS CHANNEL]:"]
        
        for msg in relevant_messages[:MAX_QUEUE_DISPLAY]:
            author_alias = self.user_profile_manager.get_user_alias(str(msg.author.id)) or msg.author.display_name
            timestamp_str = msg.created_at.astimezone(self.timezone).isoformat()
            formatted_queue.append(f"- {author_alias}: {msg.content} [{timestamp_str}]")

        if len(relevant_messages) > MAX_QUEUE_DISPLAY:
            formatted_queue.append(f"- ...and {len(relevant_messages) - MAX_QUEUE_DISPLAY} more message(s) waiting.")
        
        return "\n".join(formatted_queue)

    async def generate_response(self, message: discord.Message):
        user_id_str = str(message.author.id)
        username_str = message.author.name
        author_display_name = self.user_profile_manager.get_user_alias(user_id_str) or message.author.display_name
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Pause TinyGen as this is a resource-intensive operation
        await self.tinygen_controller.pause()
        dev_logger.info(f"Response triggered for message ID {message.id}. TinyGen processing paused.")

        interaction_id = str(message.id)
        interaction_timestamp = int(message.created_at.replace(tzinfo=dt_timezone.utc).timestamp())

        if await self._handle_alias_change(message, user_id_str, username_str, author_display_name, interaction_id, interaction_timestamp):
            await self.tinygen_controller.resume()
            return None

        # Build history once at the start of the interaction
        prompt_overhead = self.thought_processor.get_prompt_overhead_tokens("")
        history_token_budget = self.llm_max_context_tokens - prompt_overhead - self.llm_max_output_tokens
        base_history, long_term_history_str = await self.history_manager.build_llm_history(message, history_token_budget, self.message_queue, interaction_id)

        # Prepare the initial state for the reasoning loop
        current_interaction_turns = []
        user_message_content = self.thought_processor.replace_mentions_with_aliases(message.content, self.user_profile_manager)
        user_message = {"role": "user", "content": f"{author_display_name}: {user_message_content}"}
        final_response_package = None

        for i in range(self.MAX_TOOL_ITERATIONS):
            dev_logger.info(f"Tool loop iteration {i+1}/{self.MAX_TOOL_ITERATIONS} for interaction: {interaction_id}")

            # MODIFICATION: Generate the real-time queue display on each loop
            message_queue_str = self._format_message_queue_for_prompt(str(message.channel.id))
            
            message_details_for_tp = (
                f"This message is from {author_display_name} in {'DM' if is_dm else 'a public channel'}. "
                f"Your relationship with {author_display_name} is: {self.user_profile_manager.get_gen_relationship(user_id_str)}. "
            )

            # NOTE: get_next_decision in thought_processor.py must be updated to accept 'message_queue_str'
            llm_decision = await self.thought_processor.get_next_decision(
                base_history=base_history,
                current_interaction_turns=current_interaction_turns,
                user_message=user_message,
                long_term_history_str=long_term_history_str,
                message_queue_str=message_queue_str, # Pass the new queue block
                persona_data={'details_for_prompt': message_details_for_tp},
                max_tokens=self.llm_max_output_tokens
            )

            if llm_decision.get("type") == "tool_call":
                tool_name, tool_args, tool_call_id = llm_decision["name"], llm_decision["arguments"], llm_decision["id"]

                tool_execution_result = await self.thought_processor.execute_tool(tool_name, tool_args, tool_call_id, {}, user_id_str, author_display_name)
                
                summary_for_action_node = str(tool_execution_result.get("content", ""))
                if tool_name == "generate_image":
                    summary_for_action_node = tool_execution_result.get("prompt_for_logging", "No prompt found for logging.")

                if tool_name != "respond_to_user":
                    await self.neo4j_manager.insert_action(
                        interaction_id=interaction_id,
                        channel_id=str(message.channel.id),
                        action_type=tool_name,
                        timestamp=int(datetime.now(dt_timezone.utc).timestamp()),
                        reason=f"LLM tool call: {tool_name}, args: {json.dumps(tool_args)}",
                        result_summary=summary_for_action_node[:1000],
                        tool_call_id=tool_call_id
                    )

                assistant_turn = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]
                }
                tool_result_turn = {
                    "role": "tool", "tool_call_id": tool_call_id, "name": tool_name,
                    "content": str(tool_execution_result.get("content", "Tool executed with no returnable content."))
                }
                current_interaction_turns.extend([assistant_turn, tool_result_turn])

                if tool_execution_result.get("type") in ["terminal_result", "error"]:
                    final_response_package = tool_execution_result
                    break
            else:
                final_response_package = llm_decision
                break
        else:
            final_response_package = {"type": "error", "content": "Max tool iterations reached, neko."}

        if final_response_package:
            return final_response_package.get("content")

        return "I seem to be stuck in my thoughts, neko!"
