from pydantic import BaseModel
class ProposalEvaluationRequestDto(BaseModel):
    title: str
    content: str
    media_type: str
    proposal_score: int
    additional_features: dict

class ProposalEvaluationResponseDto(BaseModel):
    feedback_type: int
    feedback: str
    current_proposal_score: int
    regulation_category: list
    
class SummaryGenerationRequestDto(BaseModel):
    title: str
    content: str
    media_type: str
    proposal_score: int
    additional_features: dict
    
class SummaryGenerationResponseDto(BaseModel):
    status: bool
    title: str
    content: str
    keyword: list