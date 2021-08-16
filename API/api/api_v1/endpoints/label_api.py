from fastapi import APIRouter, Depends, status

from typing import Optional

from pydantic import BaseModel

from core.config import ALLOWED_HOSTS, PROJECT_NAME, PROJECT_VERSION, API_PORT
from core.config import DATABASE_NAME, Feedback_Label_Collection, Feedback_Template_Collection, Feedback_Suggestion_Collection, LABEL_COLLECTION

from db.mongodb import AsyncIOMotorClient, get_database
import asyncio
from typing import Any, Dict, AnyStr, List, Union
from datetime import datetime

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."

class create_new_label_body(BaseModel):
    user: str = "example@gmail.com"
    label_name: str = "Party"
    inherit: list = ["B-per", "I-per", "B-org", "I-org"]
    alias_as: list = ["String"]
    label_description: str = """Label 'Party' is to label out Party in concerto contract. Party also all the ["B-per", "I-per", "B-org", "I-org"] in current and future dataset, and when training, Party will also be label as String."""


@router.post("/label", tags = ["Optimize Data"], status_code=status.HTTP_200_OK)
async def define_new_label(data: create_new_label_body):
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]
    result = col.find({"label_name": data.label_name})
    result = await result.to_list(None)
    if len(result) != 0:
        print(f"Already have {data.label_name}")
        return {
            "message": f"Failed, Already have {data.label_name}",
            "label": result[0]
        }
    else:
        dataToStore = {
            "user": data.user,
            "label_name": data.label_name,
            "inherit": data.inherit,
            "alias_as": data.alias_as,
            "label_description": data.label_description,
            "TimeStamp": datetime.now()
        }
        try:
            result = await col.insert_one(dataToStore)
            return {
                "message": f"Success, new label {data.label_name} added."
            }
        except Exception as e:
            return {
                "message": f"Fail, please check the Error Message",
                "error_msg": str(e)
            }



class update_data_body(BaseModel):
    user: str = "example@gmail.com"
    texts: JSONArray = [
        {
            "text": "Eason",
            "labels": ["Party", "String"]
        },
        {
            "text": "will",
            "labels": ["O"]
        },
        {
            "text": "meet",
            "labels": ["O"]
        },
        {
            "text": "Dan",
            "labels": ["Party", "String"]
        },
        {
            "text": "at",
            "labels": ["O"]
        },
        {
            "text": "2021-08-04 18:00",
            "labels": ["TemporalUnit"]
        },
        {
            "text": ".",
            "labels": ["O"]
        },
    ]
    
@router.post("/data/label", tags = ["Optimize Data"], status_code=status.HTTP_200_OK)
async def update_labeled_data(data: update_data_body):
    dataToStore = {
        "user": data.user,
        "text_and_labels": data.texts,
        "TimeStamp": datetime.now(),
    }
    # todo: Check the label in text all included.
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][Feedback_Label_Collection]
    result = await col.insert_one(dataToStore)
    return "OK"



@router.post("/model/label:retrain/{label_name}", tags = ["ReTrain"], status_code=status.HTTP_200_OK)
def retrain_specific_label_model(label_name: str):
    """Re-Train Certain label with existing data."""
    dataToStore = {

    }
    return {
        "message": "success, start retrain.",
        "train-data-amount": 2500,
        "trace_id": "TBA",
    }