from pydantic import BaseModel
# Proposal


class ProposalEvaluationRequestDto(BaseModel):
    title: str
    content: str
    media_type: str
    proposal_score: float
    additional_features: dict


class FeedbackDto(BaseModel):
    feedback_type: int
    current_score: float
    comment: str
    violations: list


class SummaryDto(BaseModel):
    title: str
    content: str
    keyword: list


class ProposalEvaluationResponseDto(BaseModel):
    evaluation_result: int
    feedback: FeedbackDto
    summary: SummaryDto

# Summary


class SummaryGenerationRequestDto(BaseModel):
    title: str
    content: str
    media_type: str
    proposal_score: float
    additional_features: dict
