from fastapi import APIRouter

from channel.schemas import ChannelEvaluationRequestDto, ChannelEvaluationResponseDto
from channel.service import channel_evaluation

router = APIRouter()


@router.post("/ai/channel/evaluation")
def channel_evaluation_router(request: ChannelEvaluationRequestDto) -> ChannelEvaluationResponseDto:
    return channel_evaluation(request)
