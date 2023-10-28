from uuid import UUID
from pydantic import BaseModel,EmailStr

class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str
    user:object
    
    
    
class TokenPayload(BaseModel):
    sub: UUID = None
    exp: int = None