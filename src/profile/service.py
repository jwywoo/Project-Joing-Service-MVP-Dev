import json
from profile.schemas import ProfileEvaluationRequestDto, ProfileEvaluationResponseDto

from profile.methods.evaluation_methods import screenshot_evaluation

from profile.prompt_enum import EvaluationPrompt

def profile_evaluation(request: ProfileEvaluationRequestDto):
    url = request.url
    
    # Prompt
    profile_evaluation_prompt = EvaluationPrompt.profile_evaluation_prompt.value
    try: 
        evaluation_result = json.loads(screenshot_evaluation(url=url,prompt=profile_evaluation_prompt))
    except Exception as e:
        print(e)
        return ProfileEvaluationResponseDto(
            evaluation_status=False,
            reason="Try again later"
            )
    return ProfileEvaluationResponseDto(
        evaluation_status=evaluation_result['appropriate'],
        reason=evaluation_result['reason']
    ) 