from pydantic import BaseModel
from typing import List


class InputData(BaseModel):
    array: List[int]


class PredictionResponse(BaseModel):
    prediction: str
