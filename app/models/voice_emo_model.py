from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4
from beanie import Document, Indexed, Link, before_event, Replace, Insert
from pydantic import Field
from .user_model import User

class VoiceEmo(Document):
    v_emotion_id: UUID = Field(default_factory=uuid4, unique=True)
    status: bool = False
    emotion_type: Indexed(str)
    description: str = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    owner: Link[User]
    
    def __repr__(self) -> str:
        return f"<VoiceEmo {self.emotion_type}>"

    def __str__(self) -> str:
        return self.emotion_type

    def __hash__(self) -> int:
        return hash(self.emotion_type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VoiceEmo):
            return self.v_emotion_id == other.v_emotion_id
        return False
    
    @before_event([Replace, Insert])
    def update_update_at(self):
        self.updated_at = datetime.utcnow()
        
    
    class Settings:
        collection_name = "vemotions"