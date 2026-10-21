import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

def get_leave_balance(employee_id):
    leave_data = {
        "EMP001":12,
        "EMP002":7,
        "EMP003":15
    }

    if employee_id not in leave_data:
        return f"Employee {employee_id} is not found"   

    return leave_data[employee_id]
    
def get_employee_department(employee_id):
    department_data = {
        "EMP001":"Engineering",
        "EMP002":"HR",
        "EMP003":"Finance"
    }    

    if employee_id not in department_data:
        return f"Employee {employee_id} is not found"

    return department_data[employee_id]

def get_department_manager(department):
    manager_data = {
        "Engineering":"Rahul",
        "HR":"Priya",
        "Finance":"Arun"
    }

    if department not in manager_data:
        return f"Department {department} not found."
    return manager_data[department]

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
"required":["employee_id"]},
{
"type":"function",
"name":"get_department_manager",
"description":"Used to find manager of the department",
"parameters":{
    "type":"object",
          "properties": {
                "department": {
                    "type": "string",
                    "description": "The department name"
                }

                        }
            },
"required":["department"]}]


available_tools = {
    "get_leave_balance":get_leave_balance,
    "get_employee_department":get_employee_department,
    "get_department_manager":get_department_manager
}

user_query = input("Ask something..")

input_data = user_query

while True:
    print("\n==== LLM CALL ====")
    
    response = client.responses.create(
        model = "gpt-5-mini",
        input = input_data,
        tools = tools
    )

    tool_outputs = []

    for item in response.output:
        print(" -",item.type)

        if item.type=="function_call":
            print("Tool requested: ",item.name)
            print("Arguments: ",item.arguments)

            function = available_tools.get(item.name)

            if function is None:
                print("Unknown tool requested:", item.name)

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": f"Tool {item.name} is not available."
                })

                continue
                
            args = json.loads(item.arguments)
            
            print("Parsed arguments: ",args)

            result = function(**args)

            print("Tool result: ", result)

            tool_outputs.append(
                {
                    "type":"function_call_output",
                    "call_id":item.call_id,
                    "output": str(result)
                }
            )
    if not tool_outputs:
        print("==== Final Answer ====")
        print(response.output_text)
        break
    
    print("\nSending tool results back to the model...")

    input_data = [*response.output,*tool_outputs]
