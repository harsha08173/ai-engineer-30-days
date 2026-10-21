
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    request: str
    status: str


def review_request(state: State):
    decision = interrupt(
        f"Approve this request? {state['request']}"
    )

    if decision == "yes":
        return {"status": "Approved"}

    return {"status": "Rejected"}


builder = StateGraph(State)
builder.add_node("review", review_request)
builder.add_edge(START, "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "leave-001"}}

result1 = graph.invoke(
    {"request": "Leave for Friday", "status": "Pending"},
    config=config
)
print("Paused:", result1)

result2 = graph.invoke(
    Command(resume=input()),
    config=config
)
print("Final:", result2)
