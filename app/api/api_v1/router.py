from fastapi import APIRouter
from app.api.api_v1.handlers import (
    text_emotion,
    user,
    voice_emotion,
    face_emotion,
    heart_stress_emotion,
)
from app.api.auth.jwt import auth_router

router = APIRouter()

router.include_router(user.user_router, prefix="/users", tags=["users"])
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(
    voice_emotion.voice_emo_router, prefix="/voiceEmo", tags=["voiceEmo"]
)
router.include_router(
    text_emotion.text_emo_and_suicidal, prefix="/textEmo", tags=["textEmo"]
)
router.include_router(face_emotion.face_emo_router, prefix="/faceEmo", tags=["faceEmo"])
router.include_router(
    heart_stress_emotion.heart_stress_emo_router,
    prefix="/heartStressEmo",
    tags=["heartStressEmo"],
)
