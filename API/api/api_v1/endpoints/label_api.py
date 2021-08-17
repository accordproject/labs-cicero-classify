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
    comment: str = """This is the example cased."""


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

from db.utils import convert_mongo_id
@router.get("/label/{label_name}")
async def get_label_by_name(label_name):
    mongo_client = await get_database()
    label_define_col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]
    label_data_col = mongo_client[DATABASE_NAME][Feedback_Label_Collection]

    label = await label_define_col.find_one({"label_name": label_name})
    label = convert_mongo_id(label)

    if label == None:
        pass
        #return "404"

    texts = label_data_col.find(
        {"text_and_labels.labels": {"$in": [label_name] + label["inherit"]}},
        {"text_and_labels": False}
    )
    texts = await texts.to_list(None)
    texts_count = len(texts)

    label["data_count"] = texts_count

    label["description(auto_generated)"] = f"""Label "{label["label_name"]}" is to label out {label["label_name"]} in concerto contract. {label["label_name"]} also all the {label["inherit"]} labels in current and future dataset, and when training {label["alias_as"]}, text which labeled as Party will also be labeled positively. Currently, we have {label["data_count"]} datas contain this label in training dataset."""
    return label

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