from fastapi import APIRouter, HTTPException
from app.schemas.heart_stress_emo_schema import InputData, PredictionResponse
from app.util.heart_stress_predict_function import HeartStressDetectionFunction


heart_stress_emo_router = APIRouter()


@heart_stress_emo_router.post(
    "predict_emotion_heart_stress", summary="Heart", response_model=PredictionResponse
)
async def predict_heart_stress_emo(data: InputData):
    try:
        prediction = HeartStressDetectionFunction.predict_heart_stress(data.array)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
