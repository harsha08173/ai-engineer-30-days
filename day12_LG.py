from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END

class State(TypedDict):
    question:str
    answer:str

def classify(state:State):
    return {}

def hr_support(state:State):
    return {"answer":"HR team handles leave questions."}

def it_support(state:State):
    return {"answer":"IT team handles password questions."}

def route_question(state:State):
    question = state["question"].lower()
    if "leave" in question:
        return "hr"
    return "it"

builder = StateGraph(State)
builder.add_node("classify",classify)
builder.add_node("hr_support",hr_support)
builder.add_node("it_support",it_support)

builder.add_edge(START,"classify")
builder.add_conditional_edges(
    "classify",
    route_question,
    {
        "hr":"hr_support",
        "it":"it_support"
    }
)
builder.add_edge("hr_support",END)
builder.add_edge("it_support",END)

graph = builder.compile()

result  = graph.invoke({
    "question":"How do I reset my password",
    "answer":""
})

print(result)