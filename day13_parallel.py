# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph,START,END

# class State(TypedDict):
#     employee:str 
#     leave:str 
#     vpn:str 

# def check_leave(state:State):
#     return {"leave":"12 days remaining"}

# def check_vpn(state:State):
#     return {"vpn":"VPN active"}


# builder = StateGraph(State)
# builder.add_node("leave",check_leave)
# builder.add_node("vpn",check_vpn)

# builder.add_edge(START,"leave")
# builder.add_edge(START,"vpn")

# builder.add_edge("leave",END)
# builder.add_edge("vpn",END)

# graph= builder.compile()

# result = graph.invoke({
#     "employee":"Harsha",
#     "leave":"",
#     "vpn":""
# })

# print(result)

# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph,START,END 
# from langgraph.types import RetryPolicy

# class State(TypedDict):
#     result:str 

# def call_api(state: State):
#     print("Calling API...")
#     raise ValueError("Temporary API Failure.")

# builder = StateGraph(State)

# builder.add_node(
#     "api",
#     call_api,
#     retry_policy = RetryPolicy(max_attempts = 3,retry_on=ValueError)
# )

# builder.add_edge(START,"api")
# builder.add_edge("api",END)

# graph = builder.compile()

# print(graph.invoke({"result":""}))

from typing_extensions import TypedDict 
from langgraph.graph import StateGraph,START,END 

class State(TypedDict):
    request:str 
    category:str 
    response:str 

def classify_request(state:State):
    if "leave" in state["request"].lower():
        return {"category":"HR"}
    return {"category":"OTHER"}

def handle_hr(state:State):
    return {"response":"HR will handle your leave request."}

sub_builder = StateGraph(State)
sub_builder.add_node("classify",classify_request)
sub_builder.add_node("hr",handle_hr)

sub_builder.add_edge(START,"classify")
sub_builder.add_edge("classify","hr")
sub_builder.add_edge("hr",END)

subgraph = sub_builder.compile()

builder = StateGraph(State)
builder.add_node("employee_support",subgraph)
builder.add_edge(START,"employee_support")
builder.add_edge("employee_support",END)

graph = builder.compile()

print(graph.invoke(
    {
        "request":"I need leave",
        "category":"",
        "response":""
        
    }
))
