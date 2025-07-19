# main.py

import os
import logging
import asyncio
import textwrap
from datetime import datetime, timezone as dt_timezone
import discord
from discord import Client
from pathlib import Path
from dotenv import load_dotenv

from conversation import ConversationManager
from user_profiles import UserProfileManager
from database_manager import DatabaseManager
from knowledge_graph import Neo4jManager
from media_manager import MediaManager
from web_search import WebSearchManager
from tinygen_controller import TinyGenController

import pytz

log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app_timezone_str = os.getenv('TIMEZONE', 'UTC')
try:
    app_timezone = pytz.timezone(app_timezone_str)
except pytz.exceptions.UnknownTimeZoneError:
    app_timezone = pytz.utc
    logging.error(f"Unknown timezone '{app_timezone_str}' in .env. Defaulting to UTC.")


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz if tz else pytz.utc

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, dt_timezone.utc).astimezone(self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return str(dt)

log_dir_path = Path("data/logs")
log_dir_path.mkdir(parents=True, exist_ok=True)
current_timestamp_for_logs = datetime.now(app_timezone).strftime('%Y%m%d_%H%M%S')

def setup_logger(logger_name, filename_prefix, level=logging.INFO, tz=app_timezone):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.FileHandler(log_dir_path / f'{filename_prefix}_{current_timestamp_for_logs}.log', encoding='utf-8')
        formatter = TimezoneFormatter('%(asctime)s [%(levelname)s] %(name)s %(module)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z', tz=tz)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

thought_logger = setup_logger('thought', 'thought_log', logging.INFO, tz=app_timezone)
dev_logger = setup_logger('dev', 'dev_debug_log', logging.DEBUG, tz=app_timezone)
neo4j_logger = setup_logger('neo4j', 'neo4j_interaction_log', logging.INFO, tz=app_timezone)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

def split_message(text, limit=2000):
    return textwrap.wrap(text, width=limit, replace_whitespace=False, drop_whitespace=False, break_long_words=True, break_on_hyphens=False)

class DiscordBot(Client):
    def __init__(self, **options):
        super().__init__(intents=intents, **options)
        dev_logger.info("Initializing DiscordBot components...")

        self.neo4j_manager = Neo4jManager()
        self.user_profile_manager = UserProfileManager(self.neo4j_manager)
        self.database_manager = DatabaseManager()
        self.media_manager = MediaManager()
        self.web_search_manager = WebSearchManager()
        self.tinygen_controller = TinyGenController()
        
        self.message_queue = asyncio.Queue()

        self.conversation_manager = ConversationManager(
            user_profile_manager=self.user_profile_manager,
            neo4j_manager=self.neo4j_manager,
            media_manager=self.media_manager,
            web_search_manager=self.web_search_manager,
            database_manager=self.database_manager,
            tinygen_controller=self.tinygen_controller,
            message_queue=self.message_queue
        )
        
        self.bot_user_id_internal = os.getenv('BOT_USER_ID')
        
        self.is_processing_queue = False
        self.last_action_timestamp = datetime.now(dt_timezone.utc)

        dev_logger.info("DiscordBot core components initialized.")

    async def setup_hook(self) -> None:
        self.loop.create_task(self.idle_time_manager_task())

    async def idle_time_manager_task(self):
        await self.wait_until_ready()

        IDLE_SECONDS_THRESHOLD = 300
        CHECK_INTERVAL = 60

        while not self.is_closed():
            await asyncio.sleep(CHECK_INTERVAL)

            if self.is_processing_queue:
                continue

            time_since_last_action = (datetime.now(dt_timezone.utc) - self.last_action_timestamp).total_seconds()

            if time_since_last_action >= IDLE_SECONDS_THRESHOLD:
                dev_logger.info(f"Gen has been idle for {time_since_last_action:.0f} seconds. Running idle task.")
                
                self.is_processing_queue = True
                try:
                    status = await self.tinygen_controller.get_info()
                    if status:
                        is_active = status.get('is_processing_active', False)
                        queued_items = status.get('queued_items', 0)

                        if not is_active and queued_items > 0:
                            await self.tinygen_controller.process_queue()
                        elif not is_active:
                            await self.tinygen_controller.resume()
                    else:
                        await self.tinygen_controller.resume()
                    
                    self.last_action_timestamp = datetime.now(dt_timezone.utc)
                finally:
                    self.is_processing_queue = False

    async def on_ready(self):
        if not self.user:
            dev_logger.critical("Bot user object is not available on_ready.")
            return

        actual_bot_id_from_discord = str(self.user.id)
        actual_bot_username_from_discord = self.user.name

        dev_logger.info(f'{actual_bot_username_from_discord} (ID: {actual_bot_id_from_discord}) has connected to Discord!')

        if not self.bot_user_id_internal:
            self.bot_user_id_internal = actual_bot_id_from_discord
        elif self.bot_user_id_internal != actual_bot_id_from_discord:
            dev_logger.error("CRITICAL MISMATCH: BOT_USER_ID in .env differs from actual bot ID. USING ACTUAL ID.")
            self.bot_user_id_internal = actual_bot_id_from_discord

        if self.conversation_manager:
             self.conversation_manager.set_bot_user_id(self.bot_user_id_internal)

        if self.user_profile_manager and hasattr(self.user_profile_manager, 'set_gen_alias'):
            try:
                 self.user_profile_manager.set_gen_alias(
                     bot_user_id=self.bot_user_id_internal,
                     bot_discord_username=actual_bot_username_from_discord
                 )
            except Exception as e:
                dev_logger.error(f"Error setting Gen's alias/profile in Neo4j during on_ready: {e}", exc_info=True)

        if self.conversation_manager and hasattr(self.conversation_manager, 'set_channel_name_map'):
            channel_id_to_name_map = {}
            for guild in self.guilds:
                for channel in guild.text_channels:
                    if channel.permissions_for(guild.me).send_messages:
                        channel_id_to_name_map[str(channel.id)] = channel.name
            self.conversation_manager.set_channel_name_map(channel_id_to_name_map)

        try:
            if hasattr(self.database_manager, 'create_everything_collection') and not self.database_manager.collection:
                 self.database_manager.create_everything_collection()
            dev_logger.info("'Everything' Milvus collection status checked/initialized.")
        except Exception as e:
            dev_logger.error(f"Failed to ensure Milvus 'Everything' collection readiness: {e}", exc_info=True)

        dev_logger.info("Gen is now fully ready and operational.")

    async def on_member_join(self, member: discord.Member):
        if member.bot: return
        try:
            self.user_profile_manager.add_new_user(user_id=str(member.id), username=member.name)
        except Exception as e:
            dev_logger.error(f"Error adding new member '{member.name}' to user profiles: {e}", exc_info=True)

    async def on_message(self, message: discord.Message):
        if not self.user or message.author.id == self.user.id or (message.author.bot and str(message.author.id) != self.bot_user_id_internal):
            return

        try:
            await self.conversation_manager.store_message_for_context(message)
            await self.message_queue.put(message)
            dev_logger.debug(f"Message {message.id} queued. Queue size: {self.message_queue.qsize()}")
        except Exception as e:
            dev_logger.error(f"Failed to store or queue message {message.id}: {e}", exc_info=True)
            return

        if self.is_processing_queue:
            dev_logger.debug("Processor is already running. Message will be handled in turn.")
            return

        self.loop.create_task(self._process_message_queue())

    async def _process_message_queue(self):
        """Processes messages from the queue one by one until it's empty."""
        if self.is_processing_queue:
            return
            
        self.is_processing_queue = True
        dev_logger.info("Starting new message processing chain...")

        try:
            while not self.message_queue.empty():
                message_to_process = await self.message_queue.get()
                dev_logger.info(f"Processing message {message_to_process.id} from queue. Remaining: {self.message_queue.qsize()}")

                cm = self.conversation_manager
                is_dm = isinstance(message_to_process.channel, discord.DMChannel)
                mentioned = self.bot_user_id_internal in [str(m.id) for m in message_to_process.mentions]
                is_reply = message_to_process.reference and message_to_process.reference.resolved and str(message_to_process.reference.resolved.author.id) == self.bot_user_id_internal

                if not (is_dm or mentioned or is_reply):
                    continue

                final_response_package = None
                try:
                    async with message_to_process.channel.typing():
                        final_response_package = await cm.generate_response(message_to_process)
                except Exception as e:
                     dev_logger.error(f"Unhandled error in generate_response for message {message_to_process.id}: {e}", exc_info=True)
                     continue

                if final_response_package:
                    self.last_action_timestamp = datetime.now(dt_timezone.utc)
                    should_reply = not self.message_queue.empty()
                    
                    text_to_send, files_to_send = None, []
                    if isinstance(final_response_package, tuple):
                        text_to_send, files_to_send = final_response_package
                    elif isinstance(final_response_package, str):
                        text_to_send = final_response_package
                    
                    sent_message_object = None
                    try:
                        # --- FIX IS HERE ---
                        # Explicitly use .reply() or .send() based on the should_reply flag.
                        if text_to_send:
                            parts = split_message(str(text_to_send))
                            for i, part in enumerate(parts):
                                files = files_to_send if i == len(parts) - 1 else []
                                if should_reply and i == 0:
                                    sent_message_object = await message_to_process.reply(content=part, files=files)
                                else:
                                    current_sent_msg = await message_to_process.channel.send(content=part, files=files)
                                    if i == 0: sent_message_object = current_sent_msg
                        elif files_to_send:
                            if should_reply:
                                sent_message_object = await message_to_process.reply(files=files_to_send)
                            else:
                                sent_message_object = await message_to_process.channel.send(files=files_to_send)
                        # --- END OF FIX ---

                        if sent_message_object:
                            await cm.store_message_for_context(sent_message_object, existing_interaction_id=str(message_to_process.id))

                    except discord.errors.Forbidden:
                        dev_logger.warning(f"Forbidden: Cannot send/reply to channel {message_to_process.channel.id}.")
                    except Exception as e:
                        dev_logger.error(f"Failed to send response for message {message_to_process.id}: {e}", exc_info=True)
        finally:
            self.is_processing_queue = False
            dev_logger.info("Message processing chain complete. Processor is now idle.")

async def run_bot_async():
    dotenv_path = Path('.') / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        dev_logger.warning(".env file not found. Relying on externally set environment variables.")

    bot_token = os.getenv('DISCORD_TOKEN')
    if not bot_token:
        dev_logger.critical("CRITICAL: DISCORD_TOKEN not found in environment variables. Bot cannot start.")
        return

    bot_instance = None
    try:
        bot_instance = DiscordBot()
        await bot_instance.start(bot_token)
    except discord.LoginFailure:
        dev_logger.critical("CRITICAL: Failed to log in to Discord: Improper token has been passed.")
    except discord.PrivilegedIntentsRequired:
        dev_logger.critical("CRITICAL: Failed to start due to missing Privileged Intents.")
    except Exception as e:
        dev_logger.critical(f"CRITICAL: Bot failed to start or encountered a fatal error: {e}", exc_info=True)
    finally:
        if bot_instance and not bot_instance.is_closed():
            await bot_instance.close()
        dev_logger.info("Bot has been shut down or failed to start.")

if __name__ == "__main__":
    asyncio.run(run_bot_async())
