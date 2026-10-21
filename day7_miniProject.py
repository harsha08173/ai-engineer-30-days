from agents import Agent,Runner,RunContextWrapper,function_tool
from dotenv import load_dotenv
import asyncio


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
    
    return str(leave_data[employee_id])


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

    return str(department_data[employee_id])

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
    return str(manager_data[department])


agent = Agent(
    name="Employee Assistant",
    instructions="""
    You are an employee assistant.
    Help users with employee information.
    Use the available tools whenever you need employee data.
    If an employee is not found, clearly tell the user.
    """,
    tools=[
        get_leave_balance,
        get_employee_department,
        get_department_manager
    ]
)

async def main():
    result = await Runner.run(
        agent,
        "How many leaves does EMP002 have and who is the manager of EMP002?"
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())