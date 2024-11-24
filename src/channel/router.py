from fastapi import APIRouter

from channel.schemas import ChannelEvaluationRequestDto
from channel.service import channel_evaluation

router = APIRouter()


@router.post("/ai/channel/evaluation")
def channel_evaluation_router(request: ChannelEvaluationRequestDto):
    return channel_evaluation(request)
