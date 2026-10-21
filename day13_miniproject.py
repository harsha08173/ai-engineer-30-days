from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command

class SupportState(TypedDict):
    employee_request:str 
    category:str 
    status:str 
    response:str 
    attempts: int
    needs_clarification: bool
    approval:str


def classify_request(state:SupportState):
    request = state["employee_request"].lower()

    if "leave" in request:
        category = "HR"
    
    elif "vpn" in request or "password" in request or "access" in request:
        category = "IT"

    else:
        category = "OTHER"
    
    return {"category":category}


def handle_hr(state:SupportState):
    decision = interrupt(f"Approve this leave request? {state["employee_request"]}")
    if(decision=="yes"):
        return {
            "status":"approved",
            "response":"Leave request approved by Harsha",
            "approval":"yes"
        }
    return {
        "status":"rejected",
        "response":"Leave request rejected by Harshac",
        "approval":"no"
    }

def handle_it(state: SupportState):
    return {
        "status": "processed",
        "response": "IT team handles VPN and access requests.",
        "approval": "not_required"
    }

def handle_other(state: SupportState):
    return {
        "status": "needs_review",
        "response": "This request needs further review.",
        "approval": "not_required"
    }
    
def check_request(state:SupportState):
    if(len(state["employee_request"].split())<3):
        return {"needs_clarification":True}
    return {"needs_clarification":False}

def route_after_check(state:SupportState):
    if(state["needs_clarification"] and state["attempts"]<2):
        return "clarify"
    return "classify"

def clarify_request(state:SupportState):
    answer = input("Please provide more details: ")
    return {
        "attempts":state["attempts"]+1,
        "employee_request":answer,
        "needs_clarification":False
    }

def route_request(state:SupportState):
    return state["category"]

builder = StateGraph(SupportState)

builder.add_node("classify",classify_request)
builder.add_node("hr",handle_hr)
builder.add_node("it",handle_it)
builder.add_node("other",handle_other)
builder.add_node("check",check_request)
builder.add_node("clarify",clarify_request)


builder.add_edge(START,"check")
builder.add_conditional_edges("check",
        route_after_check,
        {
            "clarify":"clarify",
            "classify":"classify"
        })
builder.add_edge("clarify","check")


builder.add_conditional_edges("classify",
            route_request,
            {
                "HR":"hr",
                "IT":"it",
                "OTHER":"other"
            })
builder.add_edge("hr",END)
builder.add_edge("it",END)
builder.add_edge("other",END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable":{
        "thread_id":"employee-001"
    }
}

result = graph.invoke(
    {
        "employee_request":"VPN",
        "category":"",
        "status":"pending",
        "response":"",
        "attempts":0,
        "needs_clarification":False,
        "approval":"pending"
    },
    config = config
)



if "__interrupt__" in result:
    decision = input("Approve leave (Yes/No): ").lower().strip()
    result = graph.invoke(
        Command(resume=decision),
        config = config
    )

print("Final: ",result)

