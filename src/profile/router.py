from fastapi import APIRouter

from profile.schemas import ProfileEvaluationRequestDto, ProfileEvaluationResponseDto
from profile.service import profile_evaluation

router = APIRouter()

@router.post("/ai/profile/evaluation")
def proposal_evaluation_router(request: ProfileEvaluationRequestDto):
    return profile_evaluation(request)