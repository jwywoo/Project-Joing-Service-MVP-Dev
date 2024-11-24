import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI


def text_evaluation(description, prompt):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("user", "{list}")])
    chain = prompt_template | llm | StrOutputParser()
    return json.loads(chain.invoke({"list": description}))


def image_evaluation(base64_image, prompt):
    messages = [
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
    return json.loads(response.choices[0].message.content)
