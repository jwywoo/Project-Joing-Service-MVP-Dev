from proposal.methods.evaluation_methods import volume_evaluation, content_evaluation, regulation_evaluation
from proposal.schemas import ProposalEvaluationRequestDto
from .prompts.evaluation_prompt import EvaluationPrompt

SEP = "[SEP]"

def proposal_evaluation(request: ProposalEvaluationRequestDto):
    # Proposal from the user
    proposal = request.title + SEP \
            + request.content + SEP \
            + request.media_type + SEP \
            + str(request.proposal_score) + SEP \
            + str(request.additional_features)
    
    # Prompts for evaluation
    evaluation_prompt = EvaluationPrompt
    content_evaluation_prompt = evaluation_prompt.content_evaluation_prompt.value
    regulation_evaluation_prompt = evaluation_prompt.regulation_evaluation_prompt.value
    
    # Volume Check
    volume_evaluation(proposal)
    
    # Content Check
    content_evaluation(proposal, content_evaluation_prompt)
    
    # Regulation Check
    regulation_evaluation(proposal, regulation_evaluation_prompt)
    return {"Message": "Proposal Evaluation"}