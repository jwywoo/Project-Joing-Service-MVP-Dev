from fastapi import APIRouter

from proposal.schemas import ProposalEvaluationRequestDto, ProposalEvaluationResponseDto
from proposal.service import proposal_evaluation

router = APIRouter()

@router.post("/ai/proposal/evaluation", response_model=ProposalEvaluationResponseDto)
def proposal_evaluation_router(request: ProposalEvaluationRequestDto):
    return proposal_evaluation(request)

