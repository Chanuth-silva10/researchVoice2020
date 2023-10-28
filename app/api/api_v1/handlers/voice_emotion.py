from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File
from app.models.user_model import User
from app.api.deps.user_deps import get_current_user
from app.schemas.voice_emo_schema import EmotionOut, VoiceEmoCreate, VoiceEmoUpdate
from app.services.voice_emo_service import VoiceEmoService
from app.models.voice_emo_model import VoiceEmo
from app.util.voice_record_func_cnn import VoiceRecordFuncCNN
from app.config import settings

import os


voice_emo_router = APIRouter()


@voice_emo_router.get(
    "/", summary="Get past all emotions of the user", response_model=List[EmotionOut]
)
async def list(current_user: User = Depends(get_current_user)):
    return await VoiceEmoService.list_vemotions(current_user)


@voice_emo_router.post("/create", summary="Create VoiceEmo", response_model=VoiceEmo)
async def generate_v_emotion(
    data: VoiceEmoCreate, current_user: User = Depends(get_current_user)
):
    return await VoiceEmoService.generate_v_emotion(current_user, data)


@voice_emo_router.get(
    "/{v_emotion_id}",
    summary="Get a voiceEmo by voice emotion id",
    response_model=EmotionOut,
)
async def retrieve(v_emotion_id: UUID, current_user: User = Depends(get_current_user)):
    return await VoiceEmoService.retrieve_v_emotion(current_user, v_emotion_id)


@voice_emo_router.post("/predict_emotion", summary="Voice Recording")
def record_voice(audio: UploadFile = File(...)):
    file_path = os.path.join(settings.ROOT_DIR, "util", "tmp", audio.filename)
    with open(file_path, "wb") as f:
        f.write(audio.file.read())
    return VoiceRecordFuncCNN.predict_emotion(file_path)
