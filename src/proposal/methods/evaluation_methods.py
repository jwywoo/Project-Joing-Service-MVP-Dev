import os
import json
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def volume_evaluation(proposal):
    print("volume check!")
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    print(len(tokenizer.encode(proposal)))
    return (len(tokenizer.encode(proposal))) < 200


def content_evaluation(proposal, content_evaluation_prompt):
    print("content check!")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.4)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", content_evaluation_prompt), ("user", "{proposal}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    return json.loads(chain.invoke({"proposal": proposal}))


def regulation_evaluation(proposal, regulation_evaluation_prompt):
    print("regulation check!")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.4)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", regulation_evaluation_prompt), ("user", "{proposal}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    return json.loads(chain.invoke({"proposal": proposal}))
