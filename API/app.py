
from fastapi import FastAPI, Depends, status
import pymongo
from starlette.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY


from core.config import ALLOWED_HOSTS, PROJECT_NAME, PROJECT_VERSION, API_PORT
from core.config import DATABASE_NAME, Feedback_Label_Collection, Feedback_Template_Collection, Feedback_Suggestion_Collection, LABEL_COLLECTION
from core.errors import http_422_error_handler, http_error_handler
from db.mongodb_connect import close_mongo_connection, connect_to_mongo
from db.mongodb import AsyncIOMotorClient, get_database
import asyncio

from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient as MotorClient

app = FastAPI(title=PROJECT_NAME, version = PROJECT_VERSION)

if not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)


app.add_exception_handler(HTTPException, http_error_handler)
app.add_exception_handler(HTTP_422_UNPROCESSABLE_ENTITY, http_422_error_handler)



from typing import Optional

from pydantic import BaseModel

from typing import Any, Dict, AnyStr, List, Union

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]


# from api.api_v1.api import router as api_router
# app.include_router(api_router)

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."

class create_new_label_body(BaseModel):
    user: str = "example@gmail.com"
    label_name: str = "Party"
    inherit: list = ["B-per", "I-per", "B-org", "I-org"]
    alias_as: list = ["String"]
    label_description: str = """Label 'Party' is to label out Party in concerto contract. Party also all the ["B-per", "I-per", "B-org", "I-org"] in current and future dataset, and when training, Party will also be label as String."""


@app.post("/label", tags = ["Optimize Data"], status_code=status.HTTP_200_OK)
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
                "message": f"Success, new label {data.label_name} added.",
                "id": str(result.inserted_id),
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
    
@app.post("/data/label", tags = ["Optimize Data"], status_code=status.HTTP_200_OK)
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



@app.post("/model/label:retrain/{label_name}", tags = ["ReTrain"], status_code=status.HTTP_200_OK)
def retrain_specific_label_model(label_name: str):
    """Re-Train Certain label with existing data."""
    dataToStore = {

    }
    return {
        "message": "success, start retrain.",
        "train-data-amount": 2500,
        "trace_id": "TBA",
    }

class update_template_data_body(BaseModel):
    text: str = example_text
    template: str = "IP Payment"



@app.post("/data/template", tags = ["Optimize Data"], status_code=status.HTTP_200_OK)
def update_template_data(data: update_template_data_body):
    return "OK"



class update_suggestion_data_body(BaseModel):
    text: str = "Not Finish Yet"



@app.post("/data/suggestion", tags = ["Optimize Data"], status_code=status.HTTP_200_OK)
def update_suggestion_data(data: update_suggestion_data_body):
    return "OK"



@app.post("/model/suggestion:retrain", tags = ["ReTrain"], status_code=status.HTTP_200_OK)
def retrain_suggestion_model():
    print("hi")
    return {
        "message": "success, start retrain.",
        "train-data-amount": 2500,
    }



@app.post("/model/template:retrain", tags = ["ReTrain"], status_code=status.HTTP_200_OK)
def retrain_template_model():
    return {
        "message": "success AAA, start retrain.",
        "train-data-amount": 2500,
    }















if __name__ == "__main__":
    # import nest_asyncio
    # nest_asyncio.apply()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=13537)







