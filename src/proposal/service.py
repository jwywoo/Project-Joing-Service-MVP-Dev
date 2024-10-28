from proposal.methods.evaluation_methods import volume_evaluation, content_evaluation, regulation_evaluation
from proposal.methods.generation_methods import volume_feedback, content_feedback, regulation_feedback, summary_generator

from proposal.prompts.generation_prompt import GenerationPrompt
from proposal.prompts.evaluation_prompt import EvaluationPrompt

from proposal.schemas import ProposalEvaluationRequestDto, ProposalEvaluationResponseDto, SummaryGenerationRequestDto

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
    # summary_generation_prompt = generation_prompt.summary_generation_prompt.value
    generation_prompt = GenerationPrompt
    content_feedback_prompt = generation_prompt.content_feedback_prompt.value
    regulation_feedback_prompt = generation_prompt.regulation_feedback_prompt.value
    
    # Volume Check
    # volume_evaluation(proposal)
    # if (volume_evaluation(proposal=proposal)):
    #     return volume_feedback()
    
    # Content Check
    content_evaluation_result = content_evaluation(proposal, content_evaluation_prompt)
    print(content_evaluation_result)
    total_score = float(content_evaluation_result['message']) + float(content_evaluation_result['target']) + float(content_evaluation_result['relevance'])
    evaluated_proposal = "Message: " + content_evaluation_result['message'] + "Target: " + content_evaluation_result['target'] + "Relevance: " + content_evaluation_result['relevance'] + SEP + proposal 
    if (total_score < 6.0):
        print(content_feedback(content_feedback_prompt=content_feedback_prompt, proposal=evaluated_proposal))
    
    # Regulation Check
    if (regulation_evaluation(proposal, regulation_evaluation_prompt)):
        regulation_feedback(proposal=proposal, regulation_feedback_prompt=regulation_feedback_prompt)
        
    # Summary Generator
    # summary_generator(proposal=proposal, summary_generation_prompt=summary_generation_prompt)        
    
    return {"Message": "Proposal Evaluation Done"}


def summary_generation(request: SummaryGenerationRequestDto):
    # Proposal retrieved from db
    proposal = request.title + SEP \
            + request.content + SEP \
            + request.media_type + SEP \
            + str(request.proposal_score) + SEP \
            + str(request.additional_features)
            
    # Prompt
    generation_prompt = GenerationPrompt    
    summary_generation_prompt = generation_prompt.summary_generation_prompt.value
    
    # Generator Method
    summary = summary_generator(proposal=proposal, summary_generation_prompt=summary_generation_prompt)        
    print(summary)
    return {"Message": "Summary Generated"}