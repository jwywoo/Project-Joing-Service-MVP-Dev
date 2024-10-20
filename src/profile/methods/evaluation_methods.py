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
                    "type": "text", "text": "Classify whether this youtube channel is appropriate for teenagers or not"
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