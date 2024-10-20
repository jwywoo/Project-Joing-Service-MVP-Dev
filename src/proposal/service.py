from proposal.methods.evaluation_methods import volume_evaluation, content_evaluation, regulation_evaluation
from proposal.methods.generation_methods import volume_feedback, content_feedback, regulation_feedback, summary_generation

from proposal.prompts.generation_prompt import GenerationPrompt
from proposal.prompts.evaluation_prompt import EvaluationPrompt

from proposal.schemas import ProposalEvaluationRequestDto

SEP = "[SEP]"

def proposal_evaluation(request: ProposalEvaluationRequestDto):
    # Proposal from the user
    proposal = request.title + SEP \
            + request.content + SEP \
            + request.media_type + SEP \
            + str(request.proposal_score) + SEP \
            + str(request.additional_features)
    
    # Prompts 
    ## for evaluation
    evaluation_prompt = EvaluationPrompt
    content_evaluation_prompt = evaluation_prompt.content_evaluation_prompt.value
    regulation_evaluation_prompt = evaluation_prompt.regulation_evaluation_prompt.value
    
    ## for feedback generation
    generation_prompt = GenerationPrompt
    content_feedback_prompt = generation_prompt.content_feedback_prompt.value
    regulation_feedback_prompt = generation_prompt.regulation_feedback_prompt.value
    summary_generation_prompt = generation_prompt.summary_generation_prompt.value
    
    # Volume Check
    volume_evaluation(proposal)
    if (volume_evaluation(proposal=proposal)):
        volume_feedback()
    
    # Content Check
    if (content_evaluation(proposal, content_evaluation_prompt)):
        content_feedback(proposal=proposal, content_feedback_prompt=content_feedback_prompt)
    
    # Regulation Check
    if (regulation_evaluation(proposal, regulation_evaluation_prompt)):
        regulation_feedback(proposal=proposal, regulation_feedback_prompt=regulation_feedback_prompt)
        
    summary_generation(proposal=proposal, summary_generation_prompt=summary_generation_prompt)
    return {"Message": "Proposal Evaluation Done"}