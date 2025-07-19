# history_manager.py

import os
import logging
from datetime import datetime, timedelta, timezone as dt_timezone
import pytz
import asyncio
import discord

dev_logger = logging.getLogger('dev')

class HistoryManager:
    def __init__(self, neo4j_manager, user_profile_manager, thought_processor, tokenizer=None, channel_name_map={},
                 timezone=pytz.timezone(os.getenv('TIMEZONE', 'UTC')),
                 history_primary_timeframe_hours=24, history_primary_fetch_limit=150,
                 history_supplementary_timeframe_hours=6, history_supplementary_fetch_limit=50,
                 history_low_token_threshold_percent=0.6, history_fresh_db_msg_count_threshold=20,
                 minimum_budget_for_history_tokens=500, bot_user_id=None):
        self.neo4j_manager = neo4j_manager
        self.user_profile_manager = user_profile_manager
        self.thought_processor = thought_processor  # For replace_mentions_with_aliases
        self.tokenizer = tokenizer
        self.channel_name_map = channel_name_map
        self.timezone = timezone
        self.history_primary_timeframe_hours = history_primary_timeframe_hours
        self.history_primary_fetch_limit = history_primary_fetch_limit
        self.history_supplementary_timeframe_hours = history_supplementary_timeframe_hours
        self.history_supplementary_fetch_limit = history_supplementary_fetch_limit
        self.history_low_token_threshold_percent = history_low_token_threshold_percent
        self.history_fresh_db_msg_count_threshold = history_fresh_db_msg_count_threshold
        self.minimum_budget_for_history_tokens = minimum_budget_for_history_tokens
        self.bot_user_id = bot_user_id
        dev_logger.debug("HistoryManager initialized with dynamic settings.")

    async def build_llm_history(self, current_message: discord.Message, dynamic_history_token_budget: int, message_queue: asyncio.Queue, current_interaction_id: str) -> tuple[list, str]:
        """
        Builds the LLM conversation history by fetching messages and actions from Neo4j.
        Prioritizes channels, supplements if needed, formats content with aliases/channels/timestamps,
        summarizes non-respond_to_user actions, and enforces token budgets.
        Deduplicates across LTM, STM, queue, and current interaction.
        """
        if dynamic_history_token_budget < self.minimum_budget_for_history_tokens:
            dev_logger.warning(f"Dynamic history token budget ({dynamic_history_token_budget}) is less than minimum ({self.minimum_budget_for_history_tokens}). Returning empty history.")
            return [], ""

        priority_channel_ids = self._get_channel_ids_for_priority_fetch(current_message)
        all_raw_messages_dict = {}
        all_raw_actions_dict = {}  # For action summaries

        now_utc = datetime.now(dt_timezone.utc)
        priority_oldest_ts_cutoff = int((now_utc - timedelta(hours=self.history_primary_timeframe_hours)).timestamp())

        # Fetch priority messages
        if priority_channel_ids:
            messages = self.neo4j_manager.get_messages_from_channels(
                channel_ids=list(priority_channel_ids),
                oldest_timestamp_cutoff=priority_oldest_ts_cutoff,
                limit=self.history_primary_fetch_limit
            )
            for msg in messages:
                all_raw_messages_dict[msg['message_id']] = msg

        current_raw_token_sum = sum(msg.get('token_count', self._estimate_token_count(msg.get('content_stored', '')) + 5) for msg in all_raw_messages_dict.values())

        # Expand timeframe if few messages
        if len(all_raw_messages_dict) < self.history_fresh_db_msg_count_threshold and priority_channel_ids:
            messages = self.neo4j_manager.get_messages_from_channels(
                channel_ids=list(priority_channel_ids),
                oldest_timestamp_cutoff=None,
                limit=self.history_primary_fetch_limit
            )
            for msg in messages:
                if msg['message_id'] not in all_raw_messages_dict:
                    all_raw_messages_dict[msg['message_id']] = msg
            current_raw_token_sum = sum(msg.get('token_count', self._estimate_token_count(msg.get('content_stored', '')) + 5) for msg in all_raw_messages_dict.values())

        # Supplement from other channels if tokens low
        if current_raw_token_sum < (dynamic_history_token_budget * self.history_low_token_threshold_percent):
            supplementary_channel_ids = set(self.channel_name_map.keys()) - priority_channel_ids
            if supplementary_channel_ids:
                supplementary_oldest_ts_cutoff = int((now_utc - timedelta(hours=self.history_supplementary_timeframe_hours)).timestamp())
                messages = self.neo4j_manager.get_messages_from_channels(
                    channel_ids=list(supplementary_channel_ids),
                    oldest_timestamp_cutoff=supplementary_oldest_ts_cutoff,
                    limit=self.history_supplementary_fetch_limit
                )
                for msg in messages:
                    if msg['message_id'] not in all_raw_messages_dict:
                        all_raw_messages_dict[msg['message_id']] = msg

        # Fetch actions
        actions = self._fetch_actions(priority_oldest_ts_cutoff)
        for action in actions:
            all_raw_actions_dict[action['action_id']] = action

        if not all_raw_messages_dict and not all_raw_actions_dict:
            return [], ""

        # Combine and sort by timestamp
        combined_items = []
        for msg in all_raw_messages_dict.values():
            combined_items.append({'type': 'message', 'timestamp': msg['timestamp'], 'data': msg, 'id': msg['message_id']})
        for action in all_raw_actions_dict.values():
            combined_items.append({'type': 'action', 'timestamp': action['timestamp'], 'data': action, 'id': action['action_id']})
        sorted_combined = sorted(combined_items, key=lambda x: x['timestamp'])

        # Get IDs from queue and current interaction for deduplication
        queue_ids = {str(msg.id) for msg in list(message_queue._queue)}
        current_interaction_ids = {current_interaction_id}

        # Build history turns with deduplication
        final_history_turns = []
        current_llm_tokens = 0
        temp_selected_turns_reversed = []
        used_ids = set(queue_ids) | current_interaction_ids  # Start with queue and current to dedup against them

        interacting_user_alias = self.user_profile_manager.get_user_alias(str(current_message.author.id)) or "User"

        for item in reversed(sorted_combined):
            item_id = item['id']
            if item_id in used_ids:
                continue  # Skip if already in queue or current

            if item['type'] == 'message':
                raw_msg = item['data']
                content_with_aliases = self.thought_processor.replace_mentions_with_aliases(
                    raw_msg['content_stored'], self.user_profile_manager
                )
                speaker_alias = "Unknown Speaker"
                author_id = raw_msg.get('author_user_id')
                if raw_msg['role'] == 'user' and author_id:
                    speaker_alias = self.user_profile_manager.get_user_alias(author_id) or f"User ({author_id[-4:]})"
                elif raw_msg['role'] == 'assistant':
                    speaker_alias = "Gen"
                channel_display_name = "Unknown Channel"
                channel_id = raw_msg.get('channel_id')
                if raw_msg.get('is_dm'):
                    channel_display_name = f"DM-{interacting_user_alias}"
                elif channel_id and channel_id in self.channel_name_map:
                    channel_display_name = f"#{self.channel_name_map[channel_id]}"
                else:
                    channel_display_name = f"#{channel_id}"
                iso_timestamp = self._get_iso_timestamp(raw_msg['timestamp'])
                final_content = f"{speaker_alias}: {content_with_aliases} [Channel: {channel_display_name}, Timestamp: {iso_timestamp}]"
                turn_role = raw_msg['role']
            else:
                raw_action = item['data']
                if raw_action['action_type'] == 'respond_to_user':
                    continue
                summary = f"Action: {raw_action['action_type']} (Reason: {raw_action.get('reason', 'N/A')}, Result: {raw_action.get('result_summary', 'N/A')})"
                iso_timestamp = self._get_iso_timestamp(raw_action['timestamp'])
                final_content = f"System Note: {summary} [Timestamp: {iso_timestamp}]"
                turn_role = 'system'

            turn_token_count = self._estimate_token_count(final_content) + 5
            if current_llm_tokens + turn_token_count <= dynamic_history_token_budget:
                temp_selected_turns_reversed.append({"role": turn_role, "content": final_content})
                current_llm_tokens += turn_token_count
                used_ids.add(item_id)
            else:
                break

        final_stm_turns = list(reversed(temp_selected_turns_reversed))

        # LTM construction (placeholder, as omitted in prior versions)
        long_term_history_str = ""  # Return empty if no LTM

        return final_stm_turns, long_term_history_str

    def _get_iso_timestamp(self, timestamp: int):
        """Convert Unix timestamp to ISO format with timezone."""
        return datetime.fromtimestamp(timestamp, dt_timezone.utc).astimezone(self.timezone).isoformat()

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using tokenizer or fallback."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Fallback approximation

    def _fetch_actions(self, oldest_timestamp_cutoff: int = None) -> list[dict]:
        """Fetch actions from Neo4j, filtered by timestamp if provided."""
        try:
            with self.neo4j_manager.driver.session() as session:
                query = "MATCH (a:Action)"
                if oldest_timestamp_cutoff is not None:
                    query += " WHERE a.timestamp >= $oldest_timestamp_cutoff"
                query += " RETURN a.action_id AS action_id, a.action_type AS action_type, a.timestamp AS timestamp, a.reason AS reason, a.result_summary AS result_summary"
                result = session.run(query, oldest_timestamp_cutoff=oldest_timestamp_cutoff)
                return [dict(record) for record in result]
        except Exception as e:
            dev_logger.error(f"Failed to fetch actions from Neo4j: {e}", exc_info=True)
            return []

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
