import asyncio
from agents import Agent,Runner
from dotenv import load_dotenv
from agents import function_tool


load_dotenv()


@function_tool
def get_leave_balance(employee_id:str)->str:
    """Get the remaining leaves for an employee"""
    leave_data = {
        "EMP001":12,
        "EMP002":7,
        "EMP003":15
    }

    if employee_id not in leave_data:
        return f"Employee {employee_id} is not found."
    
    return leave_data[employee_id]


@function_tool
def get_employee_department(employee_id:str)->str:
    """Get the department of an employee"""
    department_data = {
        "EMP001":"Engineering",
        "EMP002":"HR",
        "EMP003":"Finance"
    }    

    if employee_id not in department_data:
        return f"Employee {employee_id} is not found"

    return department_data[employee_id]

@function_tool
def get_department_manager(department:str)->str:
    """Get the manager of the department"""
    manager_data = {
        "Engineering":"Rahul",
        "HR":"Priya",
        "Finance":"Arun"
    }

    if department not in manager_data:
        return f"Department {department} not found."
    return manager_data[department]

agent = Agent(
    name = "Employee Assistant",
    instructions= "Help Employees with questions. Use the available tools when you need information.",
    tools = [get_leave_balance,get_department_manager,get_employee_department]
)

async def main():

    question = input("Ask a question..")
    result = await Runner.run(
        agent,
        question
    )

    print(result.final_output)


if __name__=="__main__":
    asyncio.run(main())