import pymongo
import datetime
from core.config import MONGODB_URL, DATABASE_NAME, CONFIG_COLLECTION, TRAINER_LOG_COLLECTION, LABEL_TRAIN_JOB_COLLECTION

client = pymongo.MongoClient(MONGODB_URL)

config_col = client[DATABASE_NAME][CONFIG_COLLECTION]
trainer_log_col = client[DATABASE_NAME][TRAINER_LOG_COLLECTION]
training_queue_col = client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]

def config_log(name, message):
    config_col.update_one({
        "name": name,
    },{
        "$push": {"logs": {
            "message": message,
            "time": datetime.datetime.now()
        }}
    })

def trainer_log(message):
    trainer_log_col.insert_one({
        "message": message,
        "time": datetime.datetime.now()
    })

def queue_task_log(_id, message):
    training_queue_col.update_one({
            "_id": _id,
        },{
            "$push": {"logs": {
                "message": message,
                "time": datetime.datetime.now()
            }}
        })

def change_service_status(name, status):
    config_col.update_one({
        "name": name,
    },{
        "$set": {"status": status},
    })