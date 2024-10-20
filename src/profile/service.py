from profile.schemas import ProfileEvaluationRequestDto

from profile.methods.evaluation_methods import screenshot_evaluation

from profile.prompt_enum import EvaluationPrompt

def profile_evaluation(request: ProfileEvaluationRequestDto):
    url = request.url
    
    # Prompt
    profile_evaluation_prompt = EvaluationPrompt.profile_evaluation_prompt.value
    
    print(screenshot_evaluation(url=url,prompt=profile_evaluation_prompt))
    return {"Message": "Profile Evaluation Done"}