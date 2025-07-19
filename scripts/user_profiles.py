# user_profiles.py

import os
import json
import logging
from knowledge_graph import Neo4jManager

thought_logger = logging.getLogger('thought')
dev_logger = logging.getLogger('dev')

class UserProfileManager:
    def __init__(self, neo4j_manager: Neo4jManager, gen_profile_path='data/gen_profile.json'):
        self.neo4j_manager = neo4j_manager
        self.gen_profile_path = gen_profile_path
        self._initialize_gen_profile()
        self.gen_profile = self.load_gen_profile()

    def _initialize_gen_profile(self):
        """Create gen_profile.json with defaults if it doesn't exist."""
        # Ensure the 'data' directory exists
        os.makedirs(os.path.dirname(self.gen_profile_path), exist_ok=True)
        if not os.path.exists(self.gen_profile_path):
            default_gen_profile = {
                "name": "Gen", # This will be Gen's alias
                "birthdate": "1992-03-15",
                "appearance": "long red hair, emerald green eyes, steampunk style, goggles on head, corset, cogs and gears",
                "likes": ["photography", "vintage watch repair", "gaming", "anime", "urban exploring", "beer"],
                "dislikes": ["rainy days", "boring conversations"],
                "personality": "fiery, playful, angry",
                "relationships": {}, # Stores relationships with user_ids as keys
                "random_facts": ["loves saying neko!", "lives in Europe, keeps exact location private"],
                "interests": ["steampunk", "vintage watches", "photography", "coding", "urban exploring", "art", "bearded dragons", "gaming"]
            }
            try:
                with open(self.gen_profile_path, 'w', encoding='utf-8') as f:
                    json.dump(default_gen_profile, f, indent=2)
                dev_logger.info(f"Created default gen_profile.json at {self.gen_profile_path}")
            except IOError as e:
                dev_logger.error(f"IOError creating default gen_profile.json: {e}", exc_info=True)


    def load_gen_profile(self):
        """Load Gen's profile from gen_profile.json."""
        try:
            with open(self.gen_profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                dev_logger.info(f"Successfully loaded Gen's profile from {self.gen_profile_path}")
                return profile_data
        except FileNotFoundError:
            dev_logger.error(f"gen_profile.json not found at {self.gen_profile_path}. Attempting to create and use default.")
            self._initialize_gen_profile() # Create it if it's missing
            # Try loading again after creation attempt
            try:
                with open(self.gen_profile_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e_retry:
                 dev_logger.error(f"Failed to load gen_profile.json even after attempting re-creation: {e_retry}", exc_info=True)
        except json.JSONDecodeError as e:
            dev_logger.error(f"Error decoding gen_profile.json: {e}. Using a minimal default profile.", exc_info=True)
        except Exception as e:
            dev_logger.error(f"An unexpected error occurred loading gen_profile.json: {e}", exc_info=True)
        
        # Fallback to a minimal default if any error occurs during loading
        dev_logger.warning("UserProfileManager falling back to minimal default Gen profile due to loading errors.")
        return {
            "name": "Gen", "birthdate": "1992-03-15", "appearance": "default", 
            "likes": [], "dislikes": [], "personality": "default", 
            "relationships": {}, "random_facts": [], "interests": []
        }

    def add_new_user(self, user_id: str, username: str, dm_channel_id: str = None):
        """
        Add a new user to Neo4j or ensure existing user's username is current.
        user_id is the Discord User ID. username is the Discord username.
        dm_channel_id is the Discord DM Channel ID, if available.
        Initial alias is set to username.
        """
        user_id_str = str(user_id) # Ensure string type
        username_str = str(username) # Ensure string type
        dm_channel_id_str = str(dm_channel_id) if dm_channel_id is not None else None

        user_node = self.neo4j_manager.get_user(user_id_str)
        if not user_node:
            # User does not exist, create them. Neo4jManager.create_user sets alias to username by default.
            created_user = self.neo4j_manager.create_user(user_id_str, username_str, dm_channel_id_str)
            if created_user:
                thought_logger.info(f"Added new user '{username_str}' (ID: {user_id_str}) to Neo4j. DM Channel: {dm_channel_id_str or 'N/A'}.")
                return created_user.get('alias', username_str) # Return alias (which is username initially)
            else:
                dev_logger.error(f"Failed to create user node for {user_id_str} in add_new_user.")
                return username_str # Fallback to username if creation fails
        else:
            # User exists. Check if their Discord username has changed.
            # The alias remains what they've set, or their old username if never changed.
            if user_node.get('username') != username_str:
                # Update the username property on the node, but leave the alias as is unless they change it.
                # Neo4jManager.update_user_alias should handle setting both username and alias.
                self.neo4j_manager.update_user_alias(user_id_str, user_node.get('alias', username_str), username_str)
                thought_logger.info(f"Updated username for existing user '{user_node.get('alias', user_id_str)}' (ID: {user_id_str}) to '{username_str}'.")
            return user_node.get('alias', username_str)


    def set_gen_alias(self, bot_user_id: str, bot_discord_username: str):
        """
        Set/Update Gen's profile in Neo4j.
        Her user_id is bot_user_id.
        Her username is bot_discord_username (actual Discord client name).
        Her alias is taken from gen_profile.json ("name" field).
        """
        bot_user_id_str = str(bot_user_id)
        bot_discord_username_str = str(bot_discord_username)
        # Ensure gen_profile is loaded; self.gen_profile should be populated by __init__
        gen_profile_name = self.gen_profile.get("name", "Gen") # This is Gen's persona name, used as her alias

        gen_user_node = self.neo4j_manager.get_user(bot_user_id_str)

        if not gen_user_node:
            # Gen's node doesn't exist with her actual bot_user_id. Create it.
            # Neo4jManager.create_user will set user_id, username, and alias (initially to username).
            self.neo4j_manager.create_user(
                user_id=bot_user_id_str, 
                username=bot_discord_username_str, # Store her actual Discord username
                dm_channel_id=None # Gen doesn't have a DM channel with herself
            )
            thought_logger.info(f"Created Gen's user node (ID: {bot_user_id_str}, Discord Username: {bot_discord_username_str}).")
            # Now, explicitly set her alias to her profile name, ensuring username is also correct.
            self.neo4j_manager.update_user_alias(
                user_id=bot_user_id_str, 
                new_alias=gen_profile_name, 
                username=bot_discord_username_str # Ensure username is correctly set/retained
            )
            thought_logger.info(f"Set Gen's alias to '{gen_profile_name}' (from profile) for user_id '{bot_user_id_str}'.")
        else:
            # Gen's node exists. Ensure her Discord username and profile alias are up-to-date.
            # Her alias should match gen_profile.json, and username should match her current Discord username.
            if (gen_user_node.get('alias') != gen_profile_name or 
                gen_user_node.get('username') != bot_discord_username_str):
                self.neo4j_manager.update_user_alias(
                    user_id=bot_user_id_str, 
                    new_alias=gen_profile_name, 
                    username=bot_discord_username_str # Pass current Discord username
                )
                thought_logger.info(f"Updated Gen's profile in Neo4j: User ID '{bot_user_id_str}', Discord Username '{bot_discord_username_str}', Profile Alias '{gen_profile_name}'.")
            else:
                dev_logger.debug(f"Gen's profile (ID: {bot_user_id_str}) in Neo4j is already up-to-date (Discord Username: {bot_discord_username_str}, Profile Alias: {gen_profile_name}).")
        
        return gen_profile_name # Return the alias Gen should be known by

    def get_user_alias(self, user_id: str):
        """Get the preferred alias for a user by ID, falling back to username if alias is null/empty."""
        user_id_str = str(user_id)
        user = self.neo4j_manager.get_user(user_id_str)
        if user:
            alias = user.get('alias')
            # Return alias if it's not None and not an empty string, otherwise fallback to username
            return alias if alias and alias.strip() else user.get('username') 
        dev_logger.warning(f"get_user_alias: User with ID '{user_id_str}' not found in Neo4j. Cannot retrieve alias.")
        return None 

    def get_user_by_name(self, name: str):
        """Find a user by their current alias or one of their other_names, returning user_id."""
        # This method seems fine as is, assuming Neo4jManager.get_user_by_name searches correctly.
        return self.neo4j_manager.get_user_by_name(name)

    def update_user_alias(self, user_id: str, new_alias: str, username: str):
        """
        Update a user's alias. user_id is Discord User ID. username is current Discord username.
        The Neo4jManager.update_user_alias method is responsible for moving old alias to other_names
        and setting the new alias, while also ensuring the username field is updated.
        """
        user_id_str = str(user_id)
        username_str = str(username) # Current Discord username
        new_alias_str = str(new_alias)

        user = self.neo4j_manager.get_user(user_id_str)
        if not user:
            # If user doesn't exist (e.g., alias change is the first interaction), add them.
            # dm_channel_id is unknown here, so it will be None.
            self.add_new_user(user_id_str, username_str) 
            dev_logger.info(f"User {user_id_str} was not found during alias update to '{new_alias_str}'. Added new user profile with username '{username_str}' first.")
        
        # Now update the alias. Neo4jManager's update_user_alias should handle the logic
        # of setting the new alias and also updating the username property.
        self.neo4j_manager.update_user_alias(user_id_str, new_alias_str, username_str)
        thought_logger.info(f"Updated alias for user {user_id_str} to '{new_alias_str}'. Username property set/verified as '{username_str}'.")

    def update_user_dm_channel(self, user_id: str, dm_channel_id: str):
        """Update the dm_channel_id for a user. user_id is Discord User ID, dm_channel_id is Discord DM Channel ID."""
        user_id_str = str(user_id)
        dm_channel_id_str = str(dm_channel_id)
        
        # Neo4jManager.update_user_dm_channel should handle the logic of only updating if null or different.
        self.neo4j_manager.update_user_dm_channel(user_id_str, dm_channel_id_str)
        dev_logger.debug(f"Processed request to update dm_channel_id for user {user_id_str} to {dm_channel_id_str}.")

    def get_gen_relationship(self, user_id: str):
        """Get Gen's relationship description for a user from her profile."""
        user_id_str = str(user_id)
        return self.gen_profile.get("relationships", {}).get(user_id_str, "neutral")

    def get_all_user_profiles_for_mention_mapping(self) -> list[tuple[str, str]]:
        """
        Retrieves all known names (aliases, other_names, usernames) and their corresponding Discord user_ids.
        This list is sorted by name length (descending) to ensure longer names are replaced first.
        Returns a list of tuples: [(name_to_replace: str, user_id: str)].
        """
        raw_users_data = self.neo4j_manager.get_all_users_for_alias_mapping()
        # Using a set to store (name, user_id) tuples to ensure uniqueness of name-ID pairs before sorting.
        # This handles cases where a username might be the same as an alias or an other_name for the same user.
        unique_name_id_pairs = set()

        for user_data in raw_users_data:
            user_id = user_data.get('user_id')
            if not user_id:
                dev_logger.warning(f"Skipping user data for mention mapping due to missing user_id: {user_data}")
                continue # Skip users without a valid user_id

            # Add alias
            alias = user_data.get('alias')
            if alias and alias.strip():
                unique_name_id_pairs.add((alias.strip(), user_id))
            
            # Add other_names
            other_names = user_data.get('other_names', [])
            if other_names: # Ensure other_names is not None
                for name in other_names:
                    if name and name.strip():
                        unique_name_id_pairs.add((name.strip(), user_id))

            # Add username as a fallback if it's not already effectively covered by alias or other_names for this user_id
            # This ensures that if a user hasn't set a custom alias, their Discord username can still be replaced.
            username = user_data.get('username')
            if username and username.strip():
                # Check if this username for this user_id is already present (e.g., if alias is same as username)
                is_username_covered = False
                for name_val, uid_val in unique_name_id_pairs:
                    if uid_val == user_id and name_val.lower() == username.strip().lower():
                        is_username_covered = True
                        break
                if not is_username_covered:
                     unique_name_id_pairs.add((username.strip(), user_id))
        
        # Convert set to list and sort by length of the name (descending)
        # This ensures "Gen the Great" is replaced before "Gen".
        mention_map_list_sorted = sorted(list(unique_name_id_pairs), key=lambda x: len(x[0]), reverse=True)
        
        dev_logger.debug(f"Prepared {len(mention_map_list_sorted)} unique name-to-ID mappings for mention replacement, sorted by length.")
        return mention_map_list_sorted
