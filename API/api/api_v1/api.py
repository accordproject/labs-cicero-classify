
from fastapi import APIRouter





router = APIRouter()

from .endpoints.NER_label_predict import router as NER_label_predict_router
router.include_router(NER_label_predict_router)

from .endpoints.template_predict import router as template_predict_router
router.include_router(template_predict_router)

from .endpoints.suggestion_predict import router as suggestion_predict_router
router.include_router(suggestion_predict_router)


from .endpoints.label_api import router as label_api_router
router.include_router(label_api_router)

from .endpoints.label_training_queue import router as label_training_queue_router
router.include_router(label_training_queue_router)
