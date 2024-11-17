import json
from fastapi.encoders import jsonable_encoder  # type: ignore
from app.model.base import (
    ImageResult,
    ObjectPredicted,
)
from app.services import CRUDBase
from sqlalchemy.orm.session import Session


class CRUDObjectPredicted(CRUDBase):
    def __init__(self, session: Session):
        self.session: Session = session
        self.model = ObjectPredicted

    def create(self, data):
        json_body = jsonable_encoder(data)
        new_obj = self.model(**json_body)

        self.session.add(new_obj)
        self.session.commit()
        self.session.refresh(new_obj)

        return new_obj


class CRUDImageResult(CRUDBase):
    def __init__(self, session: Session):
        self.session: Session = session
        self.model = ImageResult

    # def get_by_device_id(self, device_id: str):
    #     return (
    #         self.session.query(self.model)
    #         .filter(self.model.device_id.in_([device_id]))
    #         .first()
    #     )  # type: ignore

    # def create(self, data):
    #     json_body = jsonable_encoder(data)
    #     new_obj = self.model(**json_body)

    #     self.session.add(new_obj)
    #     self.session.commit()
    #     self.session.refresh(new_obj)

    #     return new_obj
