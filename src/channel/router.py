from fastapi import APIRouter

from channel.schemas import ProfileEvaluationRequestDto, ProfileEvaluationResponseDto
from channel.service import profile_evaluation

router = APIRouter()


@router.post("/ai/profile/evaluation")
def profile_evaluation_router(request: ProfileEvaluationRequestDto):
    return profile_evaluation(request)
