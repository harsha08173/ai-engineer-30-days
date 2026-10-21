# from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.messages import ToolMessage

# @tool 
# def add_numbers(a:int,b:int) -> int:
#     """Use the add_numbers tool to calculate 10 + 20."""

#     return a+b 

# load_dotenv()

# model = ChatOpenAI(model="gpt-5-mini")

# model_with_tools = model.bind_tools([add_numbers],tool_choice="add_numbers")

# response = model_with_tools.invoke("What is 10+20?")

# print(response.tool_calls)


# # if response.tool_calls:
# #     for tool_call in response.tool_calls:
# #         if tool_call["name"]=="add_numbers":
# #             result = add_numbers.invoke(tool_call["args"])
# #             final_response = model.invoke(
# #                 f"The tool returned {result}.Give the user the final answer.")

# messages = [("human","use the add_numbers tool to calculate 10+20"),response]

# for tool_call in response.tool_calls:
#     result = add_numbers.invoke(tool_call["args"])

#     messages.append(ToolMessage(
#         content=str(result),
#         tool_call_id = tool_call["id"]
#     ))

# final_response = model.invoke(messages)
# print("Final response: ",final_response.content)


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_leave_balance(employee_id:str)->str:
    """Get the remaining leave balance of an employee"""
    leave_balances = {
        "EMP001":12,
        "EMP002":7,
        "EMP003":15
    }
    return str(leave_balances.get(employee_id,"Employee not found"))

model = ChatOpenAI(model="gpt-5-mini")

agent = create_agent(
    model=model,
    tools = [get_leave_balance],
    system_prompt = "You are an HR assistant. Use the leave balance tool when needed."
)

# response = agent.invoke({
#     "messages":[{"role":"user","content":"What is the leave balance of EMP003?"}]
# })

# print(response["messages"][-1].content)

# response = agent.invoke({
#     "messages":[{"role":"user","content":"What about the other employee?"}]
# })


conversation = []

conversation.append({"role":"user","content":"What is the leave balance of EMP001?"})
response = agent.invoke({"messages":conversation})

print(response["messages"][-1].content)

conversation = response["messages"]
conversation.append({"role":"user","content":"What about his department?"})
response = agent.invoke({"messages":conversation})

print(print(response["messages"][-1].content))