# database_manager.py

import os
import logging
import asyncio
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from embeddings import generate_embedding

thought_logger = logging.getLogger('thought')
dev_logger = logging.getLogger('dev')

class DatabaseManager:
    def __init__(self):
        self.host = "milvus-standalone"
        self.port = "19530"
        self.collection_name = "Everything"
        self.dimension = 384  # Matches all-MiniLM-L6-v2
        self.collection = None
        self.connect()
        self._reset_mdb_if_needed()
        self.create_everything_collection()

    def connect(self):
        """Connect to Milvus with retry logic."""
        retries = 5
        for attempt in range(retries):
            try:
                connections.connect(host=self.host, port=self.port)
                dev_logger.debug("Connected to Milvus")
                return
            except Exception as e:
                if attempt < retries - 1:
                    dev_logger.warning(f"Failed to connect to Milvus (attempt {attempt + 1}/{retries}): {e}")
                    asyncio.run(asyncio.sleep(5))
                else:
                    dev_logger.error(f"Failed to connect to Milvus after {retries} attempts: {e}")
                    raise

    def _reset_mdb_if_needed(self):
        """Drop all collections if RESET_MDB is True in .env."""
        reset_mdb = os.getenv('RESET_MDB', 'False').lower() == 'true'
        if reset_mdb:
            try:
                collections = utility.list_collections()
                for collection in collections:
                    utility.drop_collection(collection)
                    dev_logger.debug(f"Dropped collection: {collection}")
                thought_logger.info("All Milvus collections dropped due to RESET_MDB=True")
            except Exception as e:
                dev_logger.error(f"Failed to drop collections: {e}")
                raise

    def create_everything_collection(self):
        """Create the Everything collection if it doesn't exist."""
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields=fields, description="Universal collection for Gen's data")
            self.collection = Collection(self.collection_name, schema)
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            dev_logger.debug(f"Created Milvus collection: {self.collection_name}")
            thought_logger.info(f"Successfully created Everything collection with HNSW index")
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()
            dev_logger.debug(f"Loaded existing Milvus collection: {self.collection_name}")

    async def insert_everything(self, text_to_embed, metadata):
        """
        Insert data into the Everything collection with embedding and JSON metadata.

        Args:
            text_to_embed (str): Text to generate the embedding.
            metadata (dict): JSON-compatible dictionary with metadata (e.g., type, title, description).

        Returns:
            int: Milvus ID of the inserted entry.
        """
        if self.collection is None:
            self.create_everything_collection()
        
        embedding = await generate_embedding(text_to_embed, use_secondary=True)
        data = [{
            "embedding": embedding.tolist(),
            "metadata": metadata
        }]
        try:
            result = self.collection.insert(data)
            milvus_id = result.primary_keys[0]
            dev_logger.debug(f"Inserted data into Everything: metadata={metadata}, milvus_id={milvus_id}")
            return milvus_id
        except Exception as e:
            dev_logger.error(f"Failed to insert data into Everything: {e}")
            raise

    async def search_everything(self, query, limit=5):
        """
        Search the Everything collection for similar embeddings.

        Args:
            query (str): Text to generate the query embedding.
            limit (int): Maximum number of results to return.

        Returns:
            list: List of search results with id, distance, and metadata.
        """
        if self.collection is None:
            self.create_everything_collection()
        
        try:
            embedding = await generate_embedding(query, use_secondary=True)
            self.collection.load()
            results = self.collection.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"ef": 200}},
                limit=limit,
                output_fields=["id", "metadata"]
            )
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "metadata": hit.entity.get("metadata")
                    })
            dev_logger.debug(f"Search results for query '{query[:50]}...': {len(search_results)} items")
            return search_results
        except Exception as e:
            dev_logger.error(f"Failed to search Everything: {e}")
            raise
