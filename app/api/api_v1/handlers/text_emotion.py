from fastapi import APIRouter, Body
from app.util.text_predict_function import TextPredictionFunction

text_emo_and_suicidal = APIRouter()


@text_emo_and_suicidal.post("/predict_emotion_and_suicidal", summary="Text")
def predict_emo_and_suicidal(text: str = Body(..., embed=True)):
    return TextPredictionFunction.predict_emo_and_suicidal(text)
