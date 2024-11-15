import logging
import io, os
from sqlalchemy.orm.session import Session
from logging.config import dictConfig
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List
from app.services.image_result import CRUDImageResult, CRUDObjectPredicted

from app.core.config import LogConfig
from loguru import logger
from app.db.database import get_db

dictConfig(LogConfig().dict())
logger_ = logging.getLogger("re_water_app")
router = APIRouter()


class ObjectDetection:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = self.load_model()

    def load_model(self):
        model = YOLO("YOLO_Custom_v8m.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)

        return results


@router.get("/health")
def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "message": "OK",
            "status": True,
        },
    )


@router.post("/predict_")
async def handler_predict(image: UploadFile = File(...)):
    contents = await image.read()
    bytes_io = io.BytesIO(contents)
    image_uploaded = Image.open(bytes_io).convert("RGB")

    model = YOLO("model-ai/yolov8_ver2.pt")
    list_label = ["plastic"]
    results = model(
        source=[image_uploaded],
        conf=0.35,
        save=False,
        project="static/images",
        name="predict",
    )

    draw = ImageDraw.Draw(image_uploaded)
    logger.debug(f"==== total result: {len(results)} =====")

    font_size = 24
    font = ImageFont.load_default(size=font_size)

    for result in results:
        boxes = result.boxes
        xyxys = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = list(map(lambda x: list_label[int(x)], boxes.cls.tolist()))
        zipped_boxes = list(zip(cls_list, conf_list, xyxys))

        logger.debug(f"==== number of item in result: {len(zipped_boxes)} =====")
        for item in zipped_boxes:
            # Draw the bounding box and label
            xmin, ymin, xmax, ymax = item[2]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

            pos_text = (xmin, ymin - (font_size + 5))
            bbox = draw.textbbox(
                pos_text,
                text=item[0],
                # font=font,
                font_size=font_size,
            )
            draw.rectangle(bbox, fill="red")
            draw.text(pos_text, item[0], fill="white", font=font)

    file_path = os.path.join("static", "images", image.filename)
    image_uploaded.save(file_path, format="JPEG")
    # save into mysql
    return JSONResponse({"file_path": file_path})


@router.post("/predicts")
async def handler_predict_multi(
    images: List[UploadFile] = File(...),
    session: Session = Depends(get_db),
):
    logger.debug(f"==== total uploaded image: {len(images)} =====")

    model = YOLO("model-ai/yolov8_ver2.pt")
    list_label = ["plastic"]
    confidence = 0.4

    font_size = 24
    font = ImageFont.load_default(size=font_size)

    list_img = []
    for idx, img in enumerate(images):
        contents = await img.read()
        bytes_io = io.BytesIO(contents)
        image_uploaded = Image.open(bytes_io).convert("RGB")
        list_img.append(image_uploaded)

    results = model(source=list_img, conf=confidence, save=False)

    list_data_save = []

    for idx, result in enumerate(results):
        draw = ImageDraw.Draw(list_img[idx])
        _file_name = images[idx].filename
        boxes = result.boxes
        xyxys = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = list(map(lambda x: list_label[int(x)], boxes.cls.tolist()))
        zipped_boxes = list(zip(cls_list, conf_list, xyxys))

        logger.debug(
            f"==== total result image: {_file_name} -- : {len(zipped_boxes)} ====="
        )
        _list_obj_predict = []
        sum_accuracy = 0
        for item in zipped_boxes:
            _label = item[0]
            _confidence = item[1]
            # Draw the bounding box and label
            xmin, ymin, xmax, ymax = item[2]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

            pos_text = (xmin, ymin - (font_size + 5))
            bbox = draw.textbbox(
                pos_text,
                text=_label,
                font_size=font_size,
            )
            draw.rectangle(bbox, fill="red")
            draw.text(pos_text, _label, fill="white", font=font)

            sum_accuracy += _confidence

            _list_obj_predict.append(
                {
                    "accuracy": _confidence,
                    "label": _label,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

        _file_path = os.path.join("static", "images", _file_name)
        list_img[idx].save(_file_path, format="JPEG")

        list_data_save.append(
            {
                "file_path": _file_path,
                "file_name": _file_name,
                "object_predict": _list_obj_predict,
                "average_accuracy": sum_accuracy / len(_list_obj_predict),
            }
        )
    # save into db
    img_result_service = CRUDImageResult(session=session)
    obj_predicted_service = CRUDObjectPredicted(session=session)
    for item in list_data_save:
        _new_obj = img_result_service.create(
            {
                "file_path": item.get("file_path"),
                "file_name": item.get("file_name"),
            }
        )
        _assign_id = [
            {**it, "image_result_id": _new_obj.id} for it in item.get("object_predict")
        ]
        obj_predicted_service.create_multiple(_assign_id)

    return JSONResponse({"data": list_data_save})
