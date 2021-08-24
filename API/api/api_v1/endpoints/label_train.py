from inspect import trace
from fastapi import APIRouter, Depends, status, Response

from typing import Optional

from pydantic import BaseModel

from core.config import ALLOWED_HOSTS, PROJECT_NAME, PROJECT_VERSION, API_PORT
from core.config import DATABASE_NAME, NER_LABEL_COLLECTION, Feedback_Template_Collection, Feedback_Suggestion_Collection, LABEL_COLLECTION, LABEL_TRAIN_JOB_COLLECTION, NER_ADAPTERS_TRAINER_NAME, CONFIG_COLLECTION

from db.mongodb import AsyncIOMotorClient, get_database
import asyncio
from typing import Any, Dict, AnyStr, List, Union
from datetime import datetime
from db.utils import convert_mongo_id
from utils.trainer_communicate import set_trainer_restart_required

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

class train_specific_NER_label_body(BaseModel):
    label_name: str = ""
    epochs: int = 2
    train_data_filter: Any = {}
    

@router.post("/model/NER/labelText/jobs", tags = ["NER Label Retrain"], status_code=status.HTTP_202_ACCEPTED)
async def train_specific_NER_label(body: train_specific_NER_label_body, response: Response):
    """Train Certain label with existing data."""

    # Check if have this label name in DB
    mongo_client = await get_database()
    label_define_col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]

    label = await label_define_col.find_one({"label_name": body.label_name})
    if label == None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "message": "Failed, Can't find this label name"
        }
    label = convert_mongo_id(label)

    label_queue_col = mongo_client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]
    dataToStore = {
        "label_name": body.label_name,
        "status": "waiting",
        "epochs": body.epochs,
        "train_data_filter": body.train_data_filter,
        "store_filename": "",
        "train_data_count": -1,
        "logs": [],
        "last_update_time": datetime.now(),
        "add_time": datetime.now(),
    }
    try:
        result = await label_queue_col.insert_one(dataToStore)
        response.status_code = status.HTTP_202_ACCEPTED
        return {
            "message": f"Success, {body.label_name} now is in Training Queue.",
            "trace_id": str(result.inserted_id),
        }
    except Exception as e:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return {
            "message": f"Fail, please check the Error Message",
            "error_msg": str(e)
        }

    # Update label_name db's training Status

from bson.objectid import ObjectId
@router.get("/model/NER/labelText/jobs/{trace_id}", tags = ["NER Label Retrain"])
async def track_specific_NER_label_training_status(trace_id: str, response: Response):
    mongo_client = await get_database()
    label_queue_col = mongo_client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]
    trace_id = ObjectId(trace_id)
    train_status = await label_queue_col.find_one({"_id": trace_id},
    {"_id": False})
    response.status_code = status.HTTP_200_OK
    return train_status
    
from bson.objectid import ObjectId
@router.delete("/model/NER/labelText/jobs/{trace_id}", tags = ["NER Label Retrain"])
async def track_specific_NER_label_training_status(trace_id: str, response: Response):
    mongo_client = await get_database()
    label_queue_col = mongo_client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]
    trace_id = ObjectId(trace_id)
    result = await label_queue_col.delete_one({"_id": trace_id})
    if result.deleted_count == 1:
        response.status_code = status.HTTP_204_NO_CONTENT
        return {}
    else:
        response.status_code = status.HTTP_304_NOT_MODIFIED
        return {}

@router.get("/model/NER/labelText/jobs", tags = ["NER Label Retrain"])
async def track_all_NER_label_training_status(response: Response, status: str = 'all'):
    """# Get NER Label Training Status by train_status\ntrain_status = ["training", "waiting", "done", "all"]"""
    status_options = ["training", "waiting", "done", "all"]
    if status not in status_options:
        response.status_code = status.HTTP_405_METHOD_NOT_ALLOWED
        return {
            "message": f"status must be in one of the {status_options}, EX: '/model/NER/labelText/jobs?status=all'."
        }
    if status == "all":
        search_filter = {}
    else:
        search_filter = {"status": status}
    
    mongo_client = await get_database()
    label_queue_col = mongo_client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]
    all_train_status = label_queue_col.find(search_filter, {"logs": False})
    all_train_status = await all_train_status.to_list(None)

    def replace_mongo_id_to_trace_id(x):
        x["trace_id"] = str(x["_id"])
        del x["_id"]
        return x

    all_train_status = list(
        map(replace_mongo_id_to_trace_id, all_train_status))
    response.status_code = status.HTTP_200_OK
    return all_train_status

@router.post("/model/NER/labelText/restartTrainer", tags = ["NER Label Retrain"])
async def restart_label_trainer_now(response: Response):
    await set_trainer_restart_required(True)
    response.status_code = status.HTTP_202_ACCEPTED
    return {
        "message": "Trainer will restart now."
    }