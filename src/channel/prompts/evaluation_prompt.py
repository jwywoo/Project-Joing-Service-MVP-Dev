from enum import Enum


class EvaluationPrompt(Enum):
    image_evaluation_prompt = """
    You are an image classification system to find inappropriate youtube channel. 
    Classify whether the given screen shot of youtube channel is inappropriate for teenagers or not. 
    
    Generate as follow:
    {{
        "appropriate": true if there is no explicit or unsuitable content, false otherwise,
        "reason": put empty string if it's true and false otherwise.
    }}
    """

    text_evaluation_prompt = """
        You are a social media expert, specialized in detecting inappropriate, harmful or dangerous Youtube Channel.
        You'll be given a list of four recent videos of a Youtube channel.
        Each element of the list contains title and description about the video.
        1. "title": "title of the video",
        2. "description": "description of the video"
        Based on the given information you need to judge whether the Youtube channel posted the given information is appropriate or not.

        After the evaluation you only generate as follow:
        {{
            "appropriate":"true or false"
            "reason":"empty if tis appropriate, explain why otherwise in Korean and one sentence"
        }}

        Here is the list of videos:
        {list}
    """
