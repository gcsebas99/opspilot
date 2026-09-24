from opspilot.config import Settings
from opspilot.store.base import Store
from opspilot.store.memory import MemoryStore
from opspilot.store.mongo import MongoStore


def build_store(settings: Settings) -> Store:
    if settings.opspilot_store == "mongo":
        if not settings.mongodb_uri:
            raise ValueError("OPSPILOT_STORE=mongo requires MONGODB_URI to be set")
        return MongoStore(settings.mongodb_uri)
    return MemoryStore()
