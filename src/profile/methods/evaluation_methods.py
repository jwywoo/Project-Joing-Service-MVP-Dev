import json 

from openai import OpenAI

from profile.methods.screenshot_methods import selenium_screenshot
def screenshot_evaluation(url, prompt):
    base64_image = selenium_screenshot(url=url)
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": """"You are provided with a screenshot of a YouTube channel page. 
                    Please analyze only the visible content in the image (such as titles, thumbnails, text, or other visible elements) to classify if the channel content is appropriate based on general content standards.
                    If any inappropriate elements appear (such as explicit language, mature themes, or unsuitable visuals), specify the reason in Korean in a single sentence. If it’s suitable, return an empty reason field.
                    Generate the result in the following format:
                        {
                            "appropriate": true if there is no explicit or unsuitable content, false otherwise,
                            "reason": put empty string if it's true and false otherwise.
                        }
                        """
                    },
                {
                    "type": "image_url",
                    "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
            ],
        }
    ]
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=300,
        )
    return response.choices[0].message.content