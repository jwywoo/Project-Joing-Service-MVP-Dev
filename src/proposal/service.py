from proposal.schemas import ProposalEvaluationRequestDto


def proposal_evaluation(request: ProposalEvaluationRequestDto):
    return {"Message": "Proposal Evaluation"}