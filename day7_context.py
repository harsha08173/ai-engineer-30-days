import asyncio
from dataclasses import dataclass
from agents import Agent,Runner,RunContextWrapper,function_tool
from dotenv import load_dotenv

load_dotenv()

@dataclass
class UserContext:
    employee_id: str
    name: str

@function_tool
def get_current_employee(context:RunContextWrapper[UserContext])->str:
    """Get the employee ID of the current logged-in employee"""

    return context.context.employee_id

@function_tool
def get_current_employee_name(context: RunContextWrapper[UserContext]) -> str:
    """Get the name of the current logged-in employee."""
    return context.context.name

agent = Agent(
    name="Employee Assistant",
    instructions="Help the employee with their questions.",
    tools = [get_current_employee, get_current_employee_name]
)

async def main():
    user_context = UserContext(
        employee_id="EMP009",
        name = "Harsha"
    )

    result = await Runner.run(
        agent,
        "What is my emp id?",
        context = user_context
    )

    print(result.final_output)


if __name__=="__main__":
    asyncio.run(main())
