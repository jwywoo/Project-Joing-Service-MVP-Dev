from pydantic import BaseModel


class ProfileEvaluationRequestDto(BaseModel):
    channel_id: str


class ProfileEvaluationResponseDto(BaseModel):
    evaluation_status: bool
    reason: str
