
from fastapi import FastAPI, Depends, status
import pymongo
from starlette.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY


from core.config import ALLOWED_HOSTS, PROJECT_NAME, PROJECT_VERSION, API_PORT
from core.config import DATABASE_NAME, NER_LABEL_COLLECTION, Feedback_Template_Collection, Feedback_Suggestion_Collection, LABEL_COLLECTION, API_V1_PREFIX
from core.errors import http_422_error_handler, http_error_handler
from db.mongodb_connect import close_mongo_connection, connect_to_mongo
from db.mongodb import AsyncIOMotorClient, get_database
import asyncio

from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient as MotorClient
from db.utils import convert_mongo_id

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


from api.api_v1.api import router as api_router
app.include_router(api_router, prefix=API_V1_PREFIX)

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."

class text_label_body(BaseModel):
    text: str = example_text





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







