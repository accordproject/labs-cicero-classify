from fastapi import APIRouter, Depends, status
from utils import adapter_model
from typing import Optional

from pydantic import BaseModel

from typing import Any, Dict, AnyStr, List, Union

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."



class text_label_body(BaseModel):
    text: str = example_text



@router.post("/model/label", tags = ["Predict"], status_code=status.HTTP_200_OK)
def text_label(data: text_label_body):
    sen, pred, logits, logits_order = adapter_model.model_predict(adapter_model.model, data.text)
    out_Tokens = []
    for i, _ in enumerate(sen):
        predictions = []
        for j, _ in enumerate(logits_order):
            predictions.append({
                "type": logits_order[j],
                "confidence": logits[i][j],
            })
        predictions.sort(key = lambda x: x["confidence"], reverse=True)
        out_Tokens.append({
            "token": sen[i],
            "predictions": predictions,
        })
    return out_Tokens

@router.get("/model/label", tags = ["Status"], status_code=status.HTTP_200_OK)
def get_model_status():
    return {
        "message": "get success",
        "status": "online",
        "tags": adapter_model.all_adapter_name
    }
