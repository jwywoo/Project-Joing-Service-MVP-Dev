from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException


class ChannelEvaluationRequestDto(BaseModel):
    channel_id: str = Field(min=24, max_length=24)
    
    @field_validator("channel_id")
    def validate_channel_id(cls, value):
        if len(value) != 24:
            raise HTTPException(status_code=422, detail="유효하지 않은 형식의 채널아이디입니다.")
        return value


class ChannelEvaluationResponseDto(BaseModel):
    evaluation_status: bool
    reason: str
