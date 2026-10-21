from fastapi import FastAPI 
from pydantic import BaseModel 


app = FastAPI(title="AI travel Assistant")

class TripRequest(BaseModel):
    destination:str
     

@app.post("/plan-trip")
def plantrip(request:TripRequest):
    return {
        "destination":request.destination,
        "message":f"Ready to plan a trip to {request.destination}"
    }


