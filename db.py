import logging
from pymongo import MongoClient, errors
from config import MONGO_URI, DB_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MongoDB")

class DatabaseManager:
    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[db_name]
            self.client.admin.command('ping')
            logger.info(f"Connexion réussie à MongoDB ({db_name})")
        except errors.ConnectionFailure as e:
            logger.error(f"Échec de connexion à MongoDB: {e}")
            raise e

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def insert_one(self, collection_name: str, document: dict):
        try:
            res = self.get_collection(collection_name).insert_one(document)
            return res.inserted_id
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion dans {collection_name}: {e}")
            return None

    def insert_many(self, collection_name: str, documents: list):
        if not documents:
            return []
        try:
            res = self.get_collection(collection_name).insert_many(documents, ordered=False)
            return res.inserted_ids
        except Exception as e:
            logger.error(f"Erreur d'insertion en masse dans {collection_name}: {e}")
            return []

db_manager = DatabaseManager()