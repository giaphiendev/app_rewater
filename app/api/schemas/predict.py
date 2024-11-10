from typing import Optional, Any
from pydantic import BaseModel


class PredictResponse(BaseModel):
    status: bool
    data: Optional[Any] = None
