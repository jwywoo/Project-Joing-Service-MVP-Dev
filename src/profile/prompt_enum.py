from enum import Enum

class EvaluationPrompt(Enum):
    profile_evaluation_prompt = """
    You are an image classification system to find inappropriate youtube channel. 
    Classify whether the given screen shot of youtube channel is inappropriate for teenagers or not. 
    Return the result in a following format {'appropriate': 'True/False'}
    """