from typing import List
from uuid import UUID
from app.models.user_model import User
from app.models.voice_emo_model import VoiceEmo
from app.schemas.voice_emo_schema import VoiceEmoCreate, VoiceEmoUpdate

class VoiceEmoService:
    @staticmethod
    async def list_vemotions(user: User) -> List[VoiceEmo]:
        vemotions = await VoiceEmo.find(VoiceEmo.owner.id == user.id).to_list()
        return vemotions
    
    @staticmethod
    async def generate_v_emotion(user: User, data: VoiceEmoCreate) -> VoiceEmo:
        voiceEmo = VoiceEmo(**data.dict(), owner=user)
        return await voiceEmo.insert()
    
    @staticmethod
    async def retrieve_v_emotion(current_user: User, v_emotion_id: UUID):
        voiceEmo = await VoiceEmo.find_one(VoiceEmo.v_emotion_id == v_emotion_id, VoiceEmo.owner.id == current_user.id)
        return voiceEmo

