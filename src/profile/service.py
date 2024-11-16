import os

from dotenv import load_dotenv
from profile.schemas import ProfileEvaluationRequestDto, ProfileEvaluationResponseDto
from profile.methods.requests_methods import youtube_data_api_request, youtube_channel_request, playlist__request, image_request
from profile.methods.preprocessing_methods import response_preprocessing, image_preprocessing
from profile.methods.evaluation_methods import text_evaluation, image_evaluation
from profile.prompts.evaluation_prompt import EvaluationPrompt
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def profile_evaluation(request: ProfileEvaluationRequestDto):
    # Getting Youtube Data API Object
    youtube_data_api = youtube_data_api_request(api_key=YOUTUBE_API_KEY)

    # Getting Channel Info
    channel_response = youtube_channel_request(
        youtube_data_api=youtube_data_api,
        channel_id=request.channel_id)

    # Getting Playlist
    playlist_response = playlist__request(
        youtube_data_api=youtube_data_api,
        youtube_channel=channel_response)

    # Parsing response aka preprocessing
    videos_text_info, thumbnail_urls = response_preprocessing(
        playlist_response=playlist_response)
    
    evaluation_prompt = EvaluationPrompt
    
    # Image Evaluation
    # Image request & preprocessing
    image_response = image_request(thumbnail_urls)
    combined_image = image_preprocessing(image_response)
    
    image_evaluation_prompt = evaluation_prompt.image_evaluation_prompt.value
    try:    
        image_evaluation_result = image_evaluation(combined_image, image_evaluation_prompt)
        if (not image_evaluation_result['appropriate'] and len(image_evaluation_result['reason']) != 0):
            return ProfileEvaluationResponseDto(
                evaluation_status=False,
                reason=image_evaluation_result['reason']
            )
    except Exception as e:
        print(e)

    # Text Evaluation
    text_evaluation_prompt = evaluation_prompt.text_evaluation_prompt.value
    text_evaluation_result = text_evaluation(
        description=videos_text_info, prompt=text_evaluation_prompt)
    return ProfileEvaluationResponseDto(
        evaluation_status=text_evaluation_result['appropriate'],
        reason=text_evaluation_result['reason']
    )
