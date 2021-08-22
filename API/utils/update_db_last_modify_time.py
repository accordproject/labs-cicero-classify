from core.config import MONGODB_URL, DATABASE_NAME, CONFIG_COLLECTION
from pymongo import MongoClient
from datetime import datetime
def update_db_last_modify_time(collection_name):
    client = MongoClient(MONGODB_URL)
    col = client[DATABASE_NAME][CONFIG_COLLECTION]
    result = col.update_one({
        "collection_name": collection_name
    }, {
        "$set": {
            "last_update_time": datetime.now()
        }
    })
    if result.modified_count == 0:
        insert_result = col.insert_one({
            "collection_name": collection_name,
            "last_update_time": datetime.now()
        })
    return True