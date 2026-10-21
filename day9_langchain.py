from dotenv import load_dotenv
import os 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel


load_dotenv()

class Interviewquestion(BaseModel):
    question:str
    difficulty:str
    follow_up:str

messages = [

    SystemMessage(content="You are an AI Engineer tutor. Explain concepts simply."),
    HumanMessage(content="What is Lang chain in 3 points?")

]

# f_prompt = prompt.invoke({"topic":"LSA"})
# print(f_prompt)


prompt = PromptTemplate.from_template("Explain {topic} in simple terms for a beginner.Answer should be only 4 setences")
model = ChatOpenAI(model="gpt-5-mini")
parser = StrOutputParser()
structured_model = model.with_structured_output(Interviewquestions)
response = structured_model.invoke("Generate one begineer-level python interview question.")
print(response)
print(response.question)
print(response.difficulty)
print(response.follow_up)

# chain = prompt | model | parser
 
# response = chain.invoke({"topic":"LangChain"})
# print(response)
# print(type(response))


# response = model.invoke(messages)

# print(response.content)
