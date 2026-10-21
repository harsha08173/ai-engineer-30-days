import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()


def get_leave_balance(employee_id):
    leave_data = {
        "EMP001":12,
        "EMP002":7,
        "EMP003":15
    }
    
    return leave_data[employee_id]
    
def get_employee_department(employee_id):
    department_data = {
        "EMP001":"Engineering",
        "EMP002":"HR",
        "EMP003":"Finance"
    }    
    return department_data[employee_id]

available_tools = {
    "get_leave_balance":get_leave_balance,
    "get_employee_department":get_employee_department
}

user_query = input("Ask something..")

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)


tools = [{
"type":"function",
"name":"get_leave_balance",
"description":"Used to find the leave balance of an employee.",
"parameters":{
    "type":"object",
          "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The employee ID"
                }

                        }
            },
"required":["employee_id"]},

{
"type":"function",
"name":"get_employee_department",
"description":"Used to find the department of an employee.",
"parameters":{
    "type":"object",
          "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The employee ID"
                }

                        }
            },
"required":["employee_id"]}]

response = client.responses.create(
    model="gpt-5-mini",
    input= user_query,
    tools = tools
)




tool_outputs = []

for item in response.output:
    if item.type=="function_call":
        function = available_tools[item.name]
        args = json.loads(item.arguments)
        result = function(**args)
        tool_outputs.append({
            "type":"function_call_output",
            "call_id":item.call_id,
            "output":str(result)
        })

# print(tool_outputs)

final_response = client.responses.create(
    model="gpt-5-mini",
    input=[*response.output,*tool_outputs]
)

print(final_response.output_text)