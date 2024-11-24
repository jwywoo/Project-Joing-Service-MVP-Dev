from pydantic import BaseModel


class ChannelEvaluationRequestDto(BaseModel):
    channel_id: str


class ChannelEvaluationResponseDto(BaseModel):
    evaluation_status: bool
    reason: str
