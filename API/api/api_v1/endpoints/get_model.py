from fastapi import APIRouter, Depends, status

from typing import Optional

from pydantic import BaseModel

from typing import Any, Dict, AnyStr, List, Union

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()





@router.get("/model/template", tags = ["Status"], status_code=status.HTTP_200_OK)
def get_model_status():
    return {
        "message": "get success",
        "status": "re-training",
        "step": {
            "current": 30,
            "total": 50,
        },
        "train-data-amount": 2500,
        "affect-labels": ['Party', 'Race', 'SpecialTerm', 'TemporalUnit', 'Time', 'Timezone', 'US_States']
    }
