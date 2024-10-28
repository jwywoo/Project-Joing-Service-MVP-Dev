import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from proposal.schemas import ProposalEvaluationResponseDto

# Global VAR
PROPOSAL = "{proposal}"

def volume_feedback():
  response_dto = ProposalEvaluationResponseDto(
    feedback_type=0,
    feedback="",
    current_proposal_score=0,
    regulation_category=[]
  )
  return response_dto

def content_feedback(proposal, content_feedback_prompt):
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
  prompt_template = ChatPromptTemplate.from_messages(
    [("system", content_feedback_prompt), ("user", PROPOSAL)]
  )
  chain = prompt_template | llm | StrOutputParser()
  return chain.invoke({"proposal": proposal})

def regulation_feedback(proposal, regulation_feedback_prompt):
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
  prompt_template = ChatPromptTemplate.from_messages(
    [("system", regulation_feedback_prompt), ("user", PROPOSAL)]
  )
  chain = prompt_template | llm | StrOutputParser()
  return chain.invoke({"proposal": proposal})

def summary_generator(proposal, summary_generation_prompt):
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
  prompt_template = ChatPromptTemplate.from_messages(
    [("system", summary_generation_prompt), ("user", PROPOSAL)]
  )
  chain = prompt_template | llm | StrOutputParser()
  print(chain.invoke({"proposal": proposal}))
  return None