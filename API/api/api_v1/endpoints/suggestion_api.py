from fastapi import APIRouter, Depends, status

from typing import Optional

from pydantic import BaseModel

from core.config import ALLOWED_HOSTS, PROJECT_NAME, PROJECT_VERSION, API_PORT
from core.config import DATABASE_NAME, NER_LABEL_COLLECTION, Feedback_Template_Collection, Feedback_Suggestion_Collection, LABEL_COLLECTION

from db.mongodb import AsyncIOMotorClient, get_database
import asyncio
from typing import Any, Dict, AnyStr, List, Union
from datetime import datetime

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

class suggest_best_practice_by_QA_model_body(BaseModel):
    question: str = "Who is the buyer?"
    context: str = "Dan (the seller) Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall (the buyer) notifies Dan in writing that it is accepting the Jeep Car."

from utils.QA_model import get_QA_model_answer

@router.post("/model/suggestion")
async def suggest_best_practice_by_QA_model(data: suggest_best_practice_by_QA_model_body):
    """Example Questions:\n"What is the Item to be sell?",\n"Who is the buyer?",\n"Who is the seller?","What is the due date?"\n\n# Example Context: \n"Dan (the seller) Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall (the buyer) notifies Dan in writing that it is accepting the Jeep Car." """
    return get_QA_model_answer(data.question, data.context)
    