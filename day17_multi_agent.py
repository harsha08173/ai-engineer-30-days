from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json

class Attraction(BaseModel):
    name:str
    location:str
    reason_to_visit:str 
    
class ResearchResult(BaseModel):
    attractions:list[Attraction]

class DayPlan(BaseModel):
    day: int
    location: str
    activities: list[str]

class ItineraryResult(BaseModel):
    days: list[DayPlan]

load_dotenv()

class TravelState(TypedDict):
    destination :str 
    research:str 
    itinerary:str 
    research_attempts :int
    validation:str

model = ChatOpenAI(model="gpt-5-mini",temperature=0)

researcher_model = model.with_structured_output(ResearchResult)
planner_model = model.with_structured_output(ItineraryResult)

def researcher(state:TravelState):
    
    print("Researcher: Gathering information...")
    response = researcher_model.invoke(
    f"You are a travel research specialist.\n"
    f"Find 3 attractions in {state['destination']}.\n"
    f"For each attraction, provide its name, location, "
    f"and why tourists should visit.\n"
    f"Keep the research concise and factual."
    )
  
    return {"research":response.model_dump_json(),
            "research_attempts":state['research_attempts']+1}

def planner(state:TravelState):
    print("Planner: Creating itinerary..")
    response = planner_model.invoke(
        f"You are an expert travel itinerary planner.\n"
        f"Create a practical 3-day itinerary for "
        f"{state['destination']} using only these research findings:\n"
        f"{state['research']}\n"
        f"Organize the itinerary by day, with morning, afternoon, "
        f"and evening activities."
        )
    return {"itinerary": response.model_dump_json()}

def validator(state:TravelState):
    itinerary = json.loads(state["itinerary"])
    if(len(itinerary["days"])==3):
        print("Validation Passed")
        return {"validation":"passed"}
    print("Validation Failed")
    return {"validation":"failed"}
    

def route_after_research(state:TravelState):
    if len(state["research"].strip())>50 or state["research_attempts"]>=2:
        return "planner"
    return "researcher"

graph_builder = StateGraph(TravelState)

graph_builder.add_node("researcher",researcher)
graph_builder.add_node("planner",planner)
graph_builder.add_node("validator",validator)

graph_builder.add_edge(START,"researcher")
graph_builder.add_conditional_edges(
    "researcher",
    route_after_research,
    {"planner":"planner",
    "researcher":"researcher"}
)
graph_builder.add_edge("planner","validator")
graph_builder.add_edge("validator",END)

travel_graph = graph_builder.compile()

if __name__=="__main__":
    result = travel_graph.invoke({
        "destination":"Goa",
        "research":"",
        "itinerary":"",
        "research_attempts":0,
        "validator":""
    })
    
    # print("\nFinal itinerary: ")
    # print(result["itinerary"])
    print("\nFinal TravelState:")

    print(result)
