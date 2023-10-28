from fastapi import APIRouter, UploadFile
from app.util.face_record_func import FaceRecordFunc

face_emo_router = APIRouter()


@face_emo_router.post("/predict_emotion", summary="predict_face_emotion")
async def generate_v_emotion(file: UploadFile):
    return FaceRecordFunc.predict_emotion(file)
