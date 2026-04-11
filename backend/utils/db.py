import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

logger = logging.getLogger("truthguard.db")

class MongoDB:
    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        self.client = None
        self.db = None
        self.collection = None

    async def connect(self):
        if not self.uri:
            logger.warning("MONGO_URI not set. MongoDB persistence disabled.")
            return False
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client.get_database("truthguard")
            self.collection = self.db.get_collection("analysis_results")
            # Verify connection
            await self.client.server_info()
            logger.info("Successfully connected to MongoDB Atlas")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    async def save_result(self, result_type: str, content_hash: str, payload: dict):
        if self.collection is None:
            return False
        try:
            document = {
                "type": result_type,
                "hash": content_hash,
                "payload": payload,
                "timestamp": datetime.utcnow()
            }
            await self.collection.update_one(
                {"hash": content_hash},
                {"$set": document},
                upsert=True
            )
            logger.info(f"Saved {result_type} result to MongoDB: {content_hash}")
            return True
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            return False

# Global instance
db = MongoDB()
