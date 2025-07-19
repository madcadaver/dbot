# knowledge_graph.py

import os
import logging
from neo4j import GraphDatabase
from datetime import datetime
import pytz
from embeddings import generate_embedding
from database_manager import DatabaseManager

dev_logger = logging.getLogger('dev')
neo4j_logger = logging.getLogger('neo4j')


class Neo4jManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.timezone = pytz.timezone(os.getenv('TIMEZONE', 'UTC'))
        self.driver = None
        self.database_manager = DatabaseManager()
        self.connect()
        self.initialize_schema()

    def connect(self):
        """Connect to Neo4j with retry logic."""
        retries = 5
        for attempt in range(retries):
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                with self.driver.session() as session:
                    result = session.run("RETURN 'Neo4j is connected!' AS message")
                    message = result.single()["message"]
                    dev_logger.debug(f"Neo4j connection successful: {message}")
                    return
            except Exception as e:
                if attempt < retries - 1:
                    dev_logger.warning(f"Failed to connect to Neo4j (attempt {attempt + 1}/{retries}): {e}")
                    import time
                    time.sleep(5)
                else:
                    dev_logger.error(f"Failed to connect to Neo4j after {retries} attempts: {e}")
                    raise

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            dev_logger.debug("Neo4j connection closed")

    def initialize_schema(self):
        """Initialize the Neo4j schema with constraints and performance indexes."""
        try:
            with self.driver.session() as session:
                # Uniqueness constraints
                session.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
                neo4j_logger.info("Ensured User.user_id uniqueness constraint.")
                session.run("CREATE CONSTRAINT interaction_id_unique IF NOT EXISTS FOR (i:Interaction) REQUIRE i.id IS UNIQUE")
                neo4j_logger.info("Ensured Interaction.id uniqueness constraint.")
                session.run("CREATE CONSTRAINT message_id_unique IF NOT EXISTS FOR (m:Message) REQUIRE m.message_id IS UNIQUE")
                neo4j_logger.info("Ensured Message.message_id uniqueness constraint.")

                # NEW: Add indexes for performance on the milvus_id property
                session.run("CREATE INDEX message_milvus_id IF NOT EXISTS FOR (n:Message) ON (n.milvus_id)")
                neo4j_logger.info("Ensured index exists for Message.milvus_id.")
                session.run("CREATE INDEX topic_milvus_id IF NOT EXISTS FOR (n:Topic) ON (n.milvus_id)")
                neo4j_logger.info("Ensured index exists for Topic.milvus_id.")

                result = session.run("SHOW CONSTRAINTS")
                constraints = [record for record in result]
                if constraints:
                    dev_logger.debug("Current Neo4j constraints:")
                    for constraint in constraints:
                        dev_logger.debug(f"  - Name: {constraint['name']}, Type: {constraint['type']}, Entity: {constraint['entityType']}, Properties: {constraint['properties']}")
                else:
                    dev_logger.debug("No constraints found in Neo4j schema (this is unexpected after initialization).")
                neo4j_logger.info("Neo4j schema initialization complete.")
        except Exception as e:
            dev_logger.error(f"Failed to initialize Neo4j schema: {e}", exc_info=True)
            raise

    async def _get_milvus_id_for_text(self, text_to_embed: str, metadata: dict) -> int | None:
        """Generates embedding and inserts into Milvus, returning the Milvus ID."""
        if not text_to_embed or not text_to_embed.strip():
            dev_logger.warning("No text provided for embedding, skipping.")
            return None
        try:
            embedding_vector = await generate_embedding(text_to_embed, use_secondary=True)
            if embedding_vector is not None and embedding_vector.any():
                milvus_id = await self.database_manager.insert_everything(
                    text_to_embed=text_to_embed,
                    metadata=metadata
                )
                dev_logger.debug(f"Content embedded and stored in Milvus with ID: {milvus_id}")
                return milvus_id
            else:
                dev_logger.warning(f"Failed to generate a valid embedding for text '{text_to_embed[:50]}...'. Milvus ID will be null.")
                return None
        except Exception as e:
            dev_logger.error(f"Failed to get Milvus ID for text: '{text_to_embed[:50]}...': {e}", exc_info=True)
            return None

    async def create_node_with_embedding(self, node_type, properties, text_to_embed=None, user_id=None, interaction_id=None):
        """
        Create a node in Neo4j. If 'milvus_id' is a key in properties, it generates an embedding
        and stores the resulting Milvus ID in that property.
        """
        properties = properties.copy()
        if 'milvus_id' in properties and text_to_embed:
            milvus_metadata = properties.copy()
            if 'milvus_id' in milvus_metadata:
                del milvus_metadata['milvus_id']
            if user_id:
                milvus_metadata['user_id'] = user_id
            if interaction_id:
                milvus_metadata['interaction_id'] = interaction_id
            properties['milvus_id'] = await self._get_milvus_id_for_text(text_to_embed, milvus_metadata)
        try:
            with self.driver.session() as session:
                query = f"CREATE (n:{node_type} $properties) RETURN n, elementId(n) AS element_id"
                result_record = session.run(query, properties=properties).single()
                if result_record:
                    created_node_props = dict(result_record['n'])
                    element_id = result_record['element_id']
                    neo4j_logger.info(f"Created {node_type} node (elementId: {element_id}) in Neo4j with properties: {created_node_props}")
                    return {'properties': created_node_props, 'element_id': element_id}
                else:
                    dev_logger.error(f"Node creation for {node_type} with properties {properties} did not return a result.")
                    return None
        except Exception as e:
            dev_logger.error(f"Failed to create {node_type} node in Neo4j with properties {properties}: {e}", exc_info=True)
            raise

    def create_user(self, user_id: str, username: str, dm_channel_id: str = None):
        """Create a new user in Neo4j or ensure existing user's username is current."""
        try:
            with self.driver.session() as session:
                timestamp = int(datetime.now(self.timezone).timestamp())
                result = session.run("""
                    MERGE (u:User {user_id: $user_id})
                    ON CREATE SET u.username = $username,
                                  u.alias = $username,
                                  u.other_names = [],
                                  u.dm_channel_id = $dm_channel_id,
                                  u.created_at = $timestamp,
                                  u.last_active_channel_id = null,
                                  u.last_active_timestamp = null
                    ON MATCH SET u.username = $username
                    RETURN u
                """, user_id=user_id, username=username, dm_channel_id=dm_channel_id, timestamp=timestamp)
                user_node = result.single()['u']
                neo4j_logger.info(f"Ensured user '{username}' (ID: {user_id}) exists in Neo4j. Properties: {dict(user_node)}")
                return dict(user_node)
        except Exception as e:
            dev_logger.error(f"Failed to create/merge user {user_id} in Neo4j: {e}", exc_info=True)
            return None

    def update_user_alias(self, user_id: str, new_alias: str, username: str):
        """Update a user's alias and ensure their username property is current."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})
                    SET u.username = $username
                    SET u.other_names = CASE
                        WHEN u.alias IS NOT NULL AND u.alias <> $new_alias AND NOT u.alias IN u.other_names
                        THEN u.other_names + u.alias
                        ELSE u.other_names
                    END
                    SET u.alias = $new_alias
                    RETURN u
                """, user_id=user_id, new_alias=new_alias, username=username)
                user = result.single()
                if user:
                    neo4j_logger.info(f"Updated alias for user {user_id} to '{new_alias}'. Username set to '{username}'.")
                    return dict(user["u"])
                else:
                    dev_logger.warning(f"Attempted to update alias for non-existent user {user_id}.")
                    return None
        except Exception as e:
            dev_logger.error(f"Failed to update alias for user {user_id} in Neo4j: {e}", exc_info=True)
            return None

    def update_user_dm_channel(self, user_id: str, dm_channel_id: str):
        """Update the dm_channel_id for a user."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})
                    WHERE u.dm_channel_id IS NULL OR u.dm_channel_id <> $dm_channel_id
                    SET u.dm_channel_id = $dm_channel_id
                    RETURN u
                """, user_id=user_id, dm_channel_id=dm_channel_id)
                if result.single():
                    neo4j_logger.info(f"Updated dm_channel_id for user {user_id} to '{dm_channel_id}' in Neo4j.")
        except Exception as e:
            dev_logger.error(f"Failed to update dm_channel_id for user {user_id} in Neo4j: {e}", exc_info=True)

    def update_user_last_active_info(self, user_id: str, channel_id: str, timestamp: int):
        """Updates the last active channel and timestamp for a user."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})
                    SET u.last_active_channel_id = $channel_id,
                        u.last_active_timestamp = $timestamp
                    RETURN u.user_id
                """, user_id=user_id, channel_id=channel_id, timestamp=timestamp)
                if result.single():
                    neo4j_logger.info(f"Updated last active info for user {user_id}: channel {channel_id}, timestamp {timestamp}.")
                else:
                    dev_logger.warning(f"Could not find user {user_id} to update last active info.")
        except Exception as e:
            dev_logger.error(f"Failed to update last active info for user {user_id}: {e}", exc_info=True)

    def get_user(self, user_id: str):
        """Retrieve a user's profile from Neo4j by their Discord User ID."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (u:User {user_id: $user_id}) RETURN u", user_id=user_id)
                user_record = result.single()
                return dict(user_record["u"]) if user_record else None
        except Exception as e:
            dev_logger.error(f"Failed to retrieve user {user_id} from Neo4j: {e}", exc_info=True)
            return None

    def get_user_by_name(self, name: str):
        """Find a user by their current alias or one of their other_names."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User)
                    WHERE u.alias = $name OR $name IN u.other_names
                    RETURN u.user_id AS userId
                    LIMIT 1
                """, name=name)
                user_record = result.single()
                return user_record["userId"] if user_record else None
        except Exception as e:
            dev_logger.error(f"Failed to find user by name '{name}' in Neo4j: {e}", exc_info=True)
            return None

    async def create_interaction(self, user_id: str, interaction_id: str, timestamp: int):
        """Create an Interaction node linked to a User."""
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (i:Interaction {id: $interaction_id})
                    ON CREATE SET i.timestamp = $timestamp
                    ON MATCH SET i.timestamp = $timestamp
                    WITH i
                    MATCH (u:User {user_id: $user_id})
                    MERGE (u)-[:INITIATED]->(i)
                """, interaction_id=interaction_id, timestamp=timestamp, user_id=user_id)
            neo4j_logger.info(f"Ensured interaction {interaction_id} (ts: {timestamp}) by user {user_id} and [:INITIATED] link.")
        except Exception as e:
            dev_logger.error(f"Failed to create interaction {interaction_id}: {e}", exc_info=True)
            raise

    async def insert_action(self, interaction_id: str, channel_id: str, action_type: str, timestamp: int, reason: str = None, result_summary: str = None, tool_call_id: str = None):
        """Insert an Action node linked to an Interaction."""
        try:
            properties = {'action_type': action_type, 'timestamp': timestamp, 'channel_id': channel_id}
            if reason: properties['reason'] = reason
            if result_summary: properties['result_summary'] = result_summary
            if tool_call_id: properties['tool_call_id'] = tool_call_id
            with self.driver.session() as session:
                result = session.run("CREATE (a:Action $properties) RETURN elementId(a) AS action_node_id", properties=properties)
                action_node_id = result.single()['action_node_id']
                session.run("""
                    MATCH (i:Interaction {id: $interaction_id})
                    MATCH (a:Action) WHERE elementId(a) = $action_node_id
                    MERGE (i)-[:INCLUDES]->(a)
                """, interaction_id=interaction_id, action_node_id=action_node_id)
            log_msg = f"Inserted action '{action_type}' for interaction {interaction_id}."
            neo4j_logger.info(log_msg)
        except Exception as e:
            dev_logger.error(f"Failed to insert action '{action_type}' for interaction {interaction_id}: {e}", exc_info=True)
            raise

    async def create_message_node(self, message_id: str, author_user_id: str, interaction_id: str,
                                  channel_id: str, is_dm: bool, role: str, content_to_store: str,
                                  timestamp: int, token_count: int,
                                  length_chars: int = None, has_attachments: bool = None):
        """Creates a :Message node and its embedding, linking it to the :User and :Interaction."""
        try:
            milvus_metadata = {
                "type": "message", "message_id": message_id, "user_id": author_user_id,
                "interaction_id": interaction_id, "role": role, "channel_id": channel_id,
                "timestamp": timestamp, "token_count": token_count
            }
            milvus_id_from_db = await self._get_milvus_id_for_text(content_to_store, milvus_metadata)
            message_properties = {
                'message_id': message_id, 'author_user_id': author_user_id,
                'interaction_id': interaction_id, 'channel_id': channel_id,
                'is_dm': is_dm, 'role': role, 'content_stored': content_to_store,
                'timestamp': timestamp, 'milvus_id': milvus_id_from_db,
                'token_count': token_count,
                'length_chars': length_chars if length_chars is not None else len(content_to_store),
                'has_attachments': has_attachments if has_attachments is not None else False
            }
            with self.driver.session() as session:
                result = session.run("""
                    MERGE (msg:Message {message_id: $props.message_id})
                    ON CREATE SET msg = $props
                    ON MATCH SET msg += $props
                    WITH msg, $props.author_user_id AS authorId, $props.interaction_id AS interactionIdVal
                    MATCH (u:User {user_id: authorId})
                    MATCH (i:Interaction {id: interactionIdVal})
                    MERGE (u)-[:SENT_MESSAGE]->(msg)
                    MERGE (msg)-[:PART_OF_INTERACTION]->(i)
                    RETURN msg
                """, props=message_properties)
                created_msg_node = result.single()['msg']
                neo4j_logger.info(f"Ensured Message node '{message_id}' exists and is linked.")
                return dict(created_msg_node)
        except Exception as e:
            dev_logger.error(f"Failed to create/link message node '{message_id}': {e}", exc_info=True)
            raise

    def get_all_users_for_alias_mapping(self) -> list[dict]:
        """Fetches all users to build the name-to-ID mapping for mentions."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:User) WHERE u.user_id IS NOT NULL
                    RETURN u.user_id AS user_id, u.alias AS alias, u.username AS username, u.other_names AS other_names
                """)
                users_data = [{"user_id": r["user_id"], "alias": r["alias"], "username": r["username"], "other_names": r["other_names"] or []} for r in result]
                dev_logger.debug(f"Fetched {len(users_data)} users for alias mapping.")
                return users_data
        except Exception as e:
            dev_logger.error(f"Failed to fetch users for alias mapping: {e}", exc_info=True)
            return []

    def get_messages_from_channels(self, channel_ids: list[str], oldest_timestamp_cutoff: int = None, limit: int = 200) -> list[dict]:
        """Fetches messages from specified channels, newer than a cutoff, up to a limit."""
        if not channel_ids: return []
        try:
            with self.driver.session() as session:
                where_clauses = ["msg.channel_id IN $channel_ids"]
                if oldest_timestamp_cutoff: where_clauses.append("msg.timestamp >= $oldest_timestamp_cutoff")
                query = f"""
                    MATCH (msg:Message) WHERE {" AND ".join(where_clauses)}
                    RETURN msg.message_id AS message_id, msg.author_user_id AS author_user_id, msg.role AS role,
                           msg.content_stored AS content_stored, msg.timestamp AS timestamp, msg.token_count AS token_count,
                           msg.channel_id AS channel_id, msg.interaction_id AS interaction_id, msg.is_dm AS is_dm
                    ORDER BY msg.timestamp DESC LIMIT $limit
                """
                result = session.run(query, channel_ids=channel_ids, oldest_timestamp_cutoff=oldest_timestamp_cutoff, limit=limit)
                messages_data = [dict(record) for record in result]
                dev_logger.debug(f"Fetched {len(messages_data)} messages from {len(channel_ids)} channels.")
                return messages_data
        except Exception as e:
            dev_logger.error(f"Failed to fetch messages from channels {channel_ids}: {e}", exc_info=True)
            return []

    # NEW: Universal node retrieval by Milvus ID
    def get_nodes_by_milvus_ids(self, milvus_ids: list[int], node_label: str = None) -> list[dict]:
        """
        Fetches full node data for any node type using a list of Milvus IDs.
        If node_label is provided, it restricts the search to that specific label for performance.
        """
        if not milvus_ids:
            return []
        try:
            with self.driver.session() as session:
                if node_label:
                    safe_label = "".join(filter(str.isalnum, node_label))
                    match_clause = f"MATCH (n:{safe_label})"
                else:
                    match_clause = "MATCH (n)"
                query = f"""
                    {match_clause}
                    WHERE n.milvus_id IN $milvus_ids
                    RETURN properties(n) AS node_properties
                """
                result = session.run(query, milvus_ids=milvus_ids)
                nodes_data = [dict(record["node_properties"]) for record in result]
                dev_logger.debug(f"Fetched {len(nodes_data)} nodes from Neo4j using Milvus IDs (Label: {node_label or 'Any'}).")
                return nodes_data
        except Exception as e:
            dev_logger.error(f"Failed to fetch nodes by Milvus IDs: {e}", exc_info=True)
            return []

    def get_full_timeline_for_interactions(self, interaction_ids: list[str]) -> list[dict]:
        """Fetches all Message and Action nodes for given interaction IDs, returned as a sorted timeline."""
        if not interaction_ids: return []
        try:
            with self.driver.session() as session:
                query = """
                    MATCH (i:Interaction) WHERE i.id IN $interaction_ids
                    OPTIONAL MATCH (msg:Message)-[:PART_OF_INTERACTION]->(i)
                    WITH i, {type: 'Message', data: properties(msg)} AS event WHERE event.data IS NOT NULL
                    RETURN event
                    UNION ALL
                    MATCH (i:Interaction) WHERE i.id IN $interaction_ids
                    OPTIONAL MATCH (act:Action)<-[:INCLUDES]-(i)
                    WITH i, {type: 'Action', data: properties(act)} AS event WHERE event.data IS NOT NULL
                    RETURN event
                """
                result = session.run(query, interaction_ids=interaction_ids)
                unique_events = {}
                for record in result:
                    event_data = record.get("event")
                    if not event_data or not event_data.get("data"): continue
                    key = event_data["data"].get("message_id") or str(event_data["data"].get("timestamp")) + str(hash(event_data["data"].get("reason", "")))
                    if key not in unique_events:
                        unique_events[key] = event_data
                sorted_timeline = sorted(list(unique_events.values()), key=lambda x: x['data']['timestamp'])
                dev_logger.debug(f"Fetched {len(sorted_timeline)} unique timeline events for {len(interaction_ids)} interactions.")
                return sorted_timeline
        except Exception as e:
            dev_logger.error(f"Failed to fetch timeline for interactions: {e}", exc_info=True)
            return []

    def get_recent_interaction_ids(self, channel_ids: list[str], limit: int = 30) -> list[str]:
        """
        Fetches a list of the most recent, unique interaction IDs from a given list of channels.
        """
        if not channel_ids:
            return []
        try:
            with self.driver.session() as session:
                # This query finds all messages in the specified channels,
                # gets their unique parent interaction IDs, and returns the most recent ones.
                query = """
                    MATCH (msg:Message)-[:PART_OF_INTERACTION]->(i:Interaction)
                    WHERE msg.channel_id IN $channel_ids
                    RETURN DISTINCT i.id AS interactionId, max(i.timestamp) as lastTimestamp
                    ORDER BY lastTimestamp DESC
                    LIMIT $limit
                """
                result = session.run(query, channel_ids=channel_ids, limit=limit)
                interaction_ids = [record["interactionId"] for record in result]
                dev_logger.debug(f"Fetched {len(interaction_ids)} recent interaction IDs from {len(channel_ids)} channels.")
                return interaction_ids
        except Exception as e:
            dev_logger.error(f"Failed to fetch recent interaction IDs: {e}", exc_info=True)
            return []
