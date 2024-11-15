from typing import Any, List, Optional

from fastapi.encoders import jsonable_encoder  # type: ignore
from loguru import logger
from sqlalchemy import desc, func, select
from sqlalchemy.orm.session import Session


class CRUDBase:
    def __init__(self, model, session: Session):
        self.session: Session = session
        self.model = model

    def get(self, object_id: int):
        return (
            self.session.query(self.model)
            .filter(self.model.id.in_([object_id]))
            .first()
        )  # type: ignore

    def get_multi(self, skip: int = 0, limit: int = 100):
        return (
            self.session.query(self.model).offset(skip).limit(limit).all()
        )  # type: ignore

    def get_all(
        self,
    ):
        return self.session.query(self.model).all()

    def _get_filter_by_args(self, dic_args: dict):
        filters = []
        for key, value in dic_args.items():  # type: str, any
            if type(value) in [list, tuple]:
                filters.append(getattr(self.model, key).in_(value))
            elif type(value) in [str, int]:
                filters.append(getattr(self.model, key) == value)
        return filters

    def get_filter(self, filters: dict, order_by_created_time=False):
        params = self._get_filter_by_args(filters)
        query = self.session.query(self.model).filter(*params)
        if order_by_created_time:
            query = query.order_by(desc(self.model.created_at))
        return query.all()

    def update_multi(self, filters: dict, data: dict):
        params = self._get_filter_by_args(filters)
        query = self.session.query(self.model).filter(*params)
        query.update(data)

    def update(self, obj_id=None, obj_in={}, current_obj=None):
        """
        params:
        obj_id: int - id of instant
        obj_in: dict - the data prepare to update
        current_obj: object - instant
        """
        if current_obj:
            db_obj = current_obj
        else:
            db_obj = self.get(object_id=obj_id)
        if not db_obj:
            return None
        obj_data = jsonable_encoder(db_obj)

        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])

        self.session.add(db_obj)
        try:
            self.session.commit()
            # self.session.refresh(db_obj)
        except Exception as err:
            logger.error(err)
            self.session.rollback()
            raise Exception(err)
        else:
            return db_obj

    def create(self, data: dict):
        json_body = jsonable_encoder(data)
        new_cam = self.model(**json_body)

        self.session.add(new_cam)
        try:
            self.session.commit()
            # self.session.refresh(new_cam)
        except Exception as err:
            logger.error(err)
            self.session.rollback()
            raise Exception(err)
        else:
            return new_cam

    def create_multiple(self, list_data: List[dict]):
        list_obj = [self.model(**item) for item in list_data]
        try:
            self.session.bulk_save_objects(list_obj)
            self.session.commit()
        except Exception as err:
            logger.error(err)
            self.session.rollback()
            raise Exception(err)

    def remove(self, object_id: int):
        obj = self.get(object_id=object_id)
        self.session.delete(obj)
        try:
            self.session.commit()
        except Exception as err:
            logger.error(err)
            self.session.rollback()
            raise Exception(err)
        else:
            return True
