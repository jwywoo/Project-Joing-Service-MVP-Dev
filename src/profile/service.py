from profile.schemas import ProfileEvaluationRequestDto


def profile_evaluation(request: ProfileEvaluationRequestDto):
    return {"Message": "Profile Evaluation Done"}