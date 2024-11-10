from fastapi import APIRouter
from app.api.routers.predict import views as predict_views

router = APIRouter()

# predict
router.include_router(predict_views.router, tags=["predict model"])
