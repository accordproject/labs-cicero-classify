
from fastapi import APIRouter

from .endpoints.adapter_predict import router as adapter_predict_router
from .endpoints.get_model import router as get_model_router
from .endpoints.label_api import router as label_api_router
from .endpoints.suggestion_api import router as suggestion_api_router
router = APIRouter()

router.include_router(adapter_predict_router)
router.include_router(get_model_router)
router.include_router(label_api_router)
router.include_router(suggestion_api_router)