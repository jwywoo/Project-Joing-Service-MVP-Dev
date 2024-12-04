import os
import json
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Volume
## Got to be more than 200 hundreds tokens
def volume_evaluation(proposal):
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    print(len(tokenizer.encode(proposal)))
    return (len(tokenizer.encode(proposal))) < 200

# Content
## Content score got to be 6.5 at least
def content_evaluation(proposal, content_evaluation_prompt):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.4)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", content_evaluation_prompt), ("user", "{proposal}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    return json.loads(chain.invoke({"proposal": proposal}))

# Regulation
## 방송통신심의위원회 Standards
def regulation_evaluation(proposal, regulation_evaluation_prompt):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.4)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", regulation_evaluation_prompt), ("user", "{proposal}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    return json.loads(chain.invoke({"proposal": proposal}))
