from pydantic import BaseModel

class ProfileEvaluationRequestDto(BaseModel):
    url:str

class ProfileEvaluationResponseDto(BaseModel):
    url:str