from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END


#Define a state
class State(TypedDict):
    name: str 
    message: str 


#Define a node
def greet(state:State):
    return {"message":f"Hello, {state['name']}!"}

def uppercase_message(state:State):
    return {"message":state["message"].upper()}

builder = StateGraph(State)

builder.add_node("greet",greet)
builder.add_node("uppercase",uppercase_message)


builder.add_edge(START,"greet")
builder.add_edge("greet","uppercase")
builder.add_edge("uppercase",END)


graph = builder.compile()

result = graph.invoke({"name":"Harsha","message":""})
print(result)