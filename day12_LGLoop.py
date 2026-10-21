from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END

class State(TypedDict):
    count:int

def increment(state:State):
    return {"count":state["count"]+1}

def check_count(state:State):
    if state["count"]<19:
        return "repeat"
    return "finish"


builder = StateGraph(State)
builder.add_node("increment",increment)

builder.add_edge(START,"increment")
builder.add_conditional_edges(
    "increment",
    check_count,
    {"repeat":"increment",
     "finish":END
    }
)

graph = builder.compile()
print(graph.invoke({"count":0}))