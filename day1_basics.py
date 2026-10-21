import os
from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel

class Question(BaseModel):
    question: str


class InterviewQuestions(BaseModel):
    Beginner:list[Question]
    Intermediate:list[Question]
    Advanced:list[Question]


load_dotenv()

model = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# response = model.responses.create(
#     model = "gpt-5-mini",
#     input = "Explain what an AI agent is to a beginner in 1 lines"
# )


# print(response.output_text)


# while(True):
#     user_input = input("What you got today?\n")
#     response = model.responses.create(
#         model = "gpt-5-mini",
#         input = user_input
#     )
#     print(response.output_text)

print("Hello I'm your AI interview Question Generator.")
role = input("Enter role: ")
exp  = input("Enter experience: ")
tech = input("Enter technology: ")

prompt = f"""
You are an expert technical interviewer.

Role: {role}
Experience: {exp}
Technologies/Skills: {tech}

Generate exactly 15 interview questions:
- 5 Beginner
- 5 Intermediate
- 5 Advanced

For every question provide:
- Question


Make the questions relevant to the candidate's role, experience, and technologies.
Do not repeat questions.

Return the response grouped by Beginner, Intermediate, and Advanced.
"""
print("Please wait the response is loading..")

response = model.responses.parse(
    model = "gpt-5-mini",
    input = prompt,
    text_format = InterviewQuestions
)



questions = response.output_parsed



print("------Beginner------\n")

for question in questions.Beginner:
    print(question.question+"\n")

print("------Intermediate------\n")

for question in questions.Intermediate:
    print(question.question+"\n")

print("------Advanced------\n")

for question in questions.Advanced:
    print(question.question+"\n")