from db.mongodb import AsyncIOMotorClient, get_database
from core.config import MONGODB_URL, DATABASE_NAME, CONFIG_COLLECTION, NER_ADAPTERS_TRAINER_NAME
from datetime import datetime
async def asyncio_update_db_last_modify_time(collection_name):
    client = await get_database()
    col = client[DATABASE_NAME][CONFIG_COLLECTION]
    result = await col.update_one({
        "collection_name": collection_name
    }, {
        "$set": {
            "last_update_time": datetime.now()
        }
    })
    if result.modified_count == 0:
        insert_result = await col.insert_one({
            "collection_name": collection_name,
            "last_update_time": datetime.now()
        })
    return True

async def set_trainer_restart_required(restart_required):
    mongo_client = await get_database()
    config_col = mongo_client[DATABASE_NAME][CONFIG_COLLECTION]
    await config_col.update_one({
        "name": NER_ADAPTERS_TRAINER_NAME,
    },{
        "$set": {"restart_required": restart_required},
    })


def update_pid(name, pid):
    from pymongo import MongoClient
    mongo_client = MongoClient(MONGODB_URL)
    config_col = mongo_client[DATABASE_NAME][CONFIG_COLLECTION]
    result = config_col.update_one({
        "name": name,
    },{
        "$set": {"pid": pid},
    })
    if result.modified_count == 0:
        config_col.insert_one({
            "name": name,
            "pid": pid,
        })
    return True