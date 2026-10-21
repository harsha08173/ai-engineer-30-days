from agents import Agent,Runner,function_tool,RunContextWrapper
from dotenv import load_dotenv
from dataclasses import dataclass
import asyncio
load_dotenv()


@dataclass
class UserContext:
    employee_id:str
    name:str


@function_tool
def get_leave_balance(context: RunContextWrapper[UserContext],
    employee_id: str = "")->str:
    """Get the remaining leaves for an employee"""
    leave_data = {
        "EMP001":12,
        "EMP002":7,
        "EMP003":15
    }

    if not employee_id:
        employee_id = context.context.employee_id

    if employee_id not in leave_data:
        return f"Employee {employee_id} is not found."
    
    return str(leave_data[employee_id])


@function_tool
def get_vpn_status(context:RunContextWrapper[UserContext],employee_id:str="")->str:
    """Get the VPN status of an employee"""
    vpn_data = {
        "EMP001":"Active",
        "EMP002":"Pending approval from LM.",
        "EMP003":"In-active"
    }

    if employee_id=="":
        employee_id = context.context.employee_id

    if(employee_id not in vpn_data):
        return f"{employee_id} vpn data is not found."
    
    return vpn_data[employee_id]

hr_agent = Agent(
    name="HR Specialist",
    instructions="""
    You are a HR Specialist.
    Help employees with HR related questions such as 
    leave,benefits, and employee policies.
    
    When the user asks about their own information using
    words such as "my" or "I", use the current employee
    from the provided context.

    When the user explicitly provides another employee ID,
    use that employee ID instead.
    """,
    tools = [get_leave_balance]
)

it_agent = Agent(
    name="IT Specialist",
    instructions="""
    You are an IT Specialist.
    Help employees with IT related questions such as
    passwords, VPN, and technical access.

    When the user asks about their own information using
    words such as "my" or "I", use the current employee
    from the provided context.

    When the user explicitly provides another employee ID,
    use that employee ID instead.

    """,
    tools = [get_vpn_status]
)

employee_agent = Agent(
    name ="Employee Assistant",
    instructions="""
    You are the main employee assistant.
    Help employees with their questions.

    Hand off IT related questions to IT specialist.
    Hand off HR related questions to HR specialist.

    If the question is not related to HR or IT,
    politely explain that you can only help with 
    employee HR and IT matters.
    """,
    handoffs = [hr_agent,it_agent]
)

async def main():
    user_context = UserContext(
    employee_id="EMP002",
    name="Harsha"
)
    result = await Runner.run(
        employee_agent,
        "What is EMP999's VPN status?",
        context=user_context
    )    

    print(result.final_output)

if __name__=="__main__":

    asyncio.run(main())