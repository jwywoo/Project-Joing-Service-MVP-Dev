from pydantic import BaseModel

class ProfileEvaluationRequestDto(BaseModel):
    url:str

class ProfileEvaluationResponseDto(BaseModel):
    evaluation_status : bool
    reason : str