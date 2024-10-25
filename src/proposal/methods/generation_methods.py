from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def volume_feedback():
  print("""
    마! 기획이 장난이가!
    양이 이게 뭐꼬?
    좀 더 채워와라 임마!
  """)
  return None

def content_feedback(proposal, content_feedback_prompt):
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
  prompt_template = ChatPromptTemplate.from_messages(
    [("system", content_feedback_prompt), ("user", "{proposal}")]
  )
  chain = prompt_template | llm | StrOutputParser()
  print(chain.invoke({"proposal": proposal}))
  return None

def regulation_feedback(proposal, regulation_feedback_prompt):
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
  prompt_template = ChatPromptTemplate.from_messages(
    [("system", regulation_feedback_prompt), ("user", "{proposal}")]
  )
  chain = prompt_template | llm | StrOutputParser()
  print(chain.invoke({"proposal": proposal}))
  return None

def summary_generator(proposal, summary_generation_prompt):
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
  prompt_template = ChatPromptTemplate.from_messages(
    [("system", summary_generation_prompt), ("user", "{proposal}")]
  )
  chain = prompt_template | llm | StrOutputParser()
  print(chain.invoke({"proposal": proposal}))
  return None