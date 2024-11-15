from typing import Any
from sqlalchemy import text, Column as SAColumn
from sqlalchemy.dialects.mysql import TIMESTAMP
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import BigInteger, Integer, String, DateTime, Text, JSON, Float
from sqlalchemy.orm import relationship

from app.db.database import Base


class Column(SAColumn):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("nullable", False)
        super().__init__(*args, **kwargs)


class TimestampMixin:
    created_at = Column(
        TIMESTAMP, nullable=False, server_default=text("current_timestamp")
    )
    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("current_timestamp on update current_timestamp"),
    )


class ImageResult(Base, TimestampMixin):
    __tablename__ = "image_result"

    id = Column(Integer, primary_key=True, unique=True, autoincrement=True)
    file_path = Column(String(255), nullable=True)
    file_name = Column(String(255), nullable=True)
    # average_accuracy = Column(Float, nullable=True, default=0.0)


class ObjectPredicted(Base, TimestampMixin):
    __tablename__ = "object_predicted"

    id = Column(BigInteger, primary_key=True, unique=True, autoincrement=True)
    accuracy = Column(Float, nullable=True, default=0.0)
    label = Column(String(255), nullable=True)
    xmin = Column(Float, nullable=True, default=0.0)
    ymin = Column(Float, nullable=True, default=0.0)
    xmax = Column(Float, nullable=True, default=0.0)
    ymax = Column(Float, nullable=True, default=0.0)

    image_result_id = Column(Integer, ForeignKey("image_result.id", ondelete="CASCADE"))

    image_result = relationship(
        ImageResult,
        primaryjoin="foreign(ObjectPredicted.image_result_id) == ImageResult.id",
    )
