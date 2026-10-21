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

tools = [
    {
        "type": "function",
        "name": "get_leave_balance",
        "description": "Get the remaining leave balance for an employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The employee ID"
                }
            },
            "required": ["employee_id"]
        }
    },    {
        "type": "function",
        "name": "get_employee_department",
        "description": "Get the employee department for an employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The employee ID"
                }
            },
            "required": ["employee_id"]
        }
    }
]

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

response = client.responses.create(
    model="gpt-5-mini",
    input= "How many leaves does EMP002 have",
    tools= tools
)


# print(response.output)

# for item in response.output:
#     print(item)

# for item in response.output:
#     if item.type=="function_call":
#         arguments = json.loads(item.arguments)
        
#         result = get_leave_balance(arguments["employee_id"])

#         print("Tool result:",result)

tool_call = None

for item in response.output:
    if item.type == "function_call":
        tool_call = item
        break

arguments = json.loads(tool_call.arguments)

function = available_tools[tool_call.name]

result = function(**arguments)

print("Tool result:", result)

tool_output = {
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
}

final_response = client.responses.create(
    model="gpt-5-mini",
    input=[  *response.output,
                tool_output
    ]
)

print(final_response.output_text)