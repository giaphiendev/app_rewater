import logging
import io
from logging.config import dictConfig

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont

from app.core.config import LogConfig
from loguru import logger

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


# @router.get("/example")
# def get_example():
#     path_file = "./data-train/images/DJI_0255.jpg"
#     # model = YOLO("model-ai/YOLO_Custom_v8m.pt")
#     model = YOLO("model-ai/YOLO_v8_1.pt")
#     list_label = ["plastic"]
#     results = model(
#         source=path_file, conf=0.8, save=False, project="runs/detect", name="predict"
#     )  # save_dir="./runs/detect/predict")

#     zipped_boxes = []
#     for result in results:
#         boxes = result.boxes
#         conf_list = boxes.conf.tolist()
#         cls_list = list(map(lambda x: list_label[int(x)], boxes.cls.tolist()))
#         zipped_boxes = list(zip(cls_list, conf_list))
#     return JSONResponse(
#         status_code=200,
#         content={"status": True, "data": zipped_boxes},
#     )


@router.post("/predict")
async def handler_predict(image: UploadFile = File(...)):
    contents = await image.read()
    bytes_io = io.BytesIO(contents)
    image_uploaded = Image.open(bytes_io).convert("RGB")

    # model = YOLO("model-ai/YOLO_v8_1.pt")
    model = YOLO("model-ai/YOLO_Custom_v8m.pt")
    list_label = ["plastic"]
    results = model(
        source=image_uploaded,
        conf=0.8,
        save=False,
        project="runs/detect",
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

    # Prepare the image for streaming
    buf = io.BytesIO()
    image_uploaded.save(buf, format="JPEG")
    buf.seek(0)

    # Return the image as a StreamingResponse
    return StreamingResponse(buf, media_type="image/jpeg")
