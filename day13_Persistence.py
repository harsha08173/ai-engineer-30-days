from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    count: int

def increment(state: State):
    return {"count": state["count"] + 1}

builder = StateGraph(State)
builder.add_node("increment", increment)

builder.add_edge(START, "increment")
builder.add_edge("increment", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo-1"}}

result1 = graph.invoke({"count": 0}, config=config)
print("First run:", result1)

print("Saved state:", graph.get_state(config).values)

result2 = graph.invoke({"count": 10}, config=config)
print("Second run:", result2)