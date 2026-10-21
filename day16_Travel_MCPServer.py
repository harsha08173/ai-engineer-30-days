from mcp.server.fastmcp import FastMCP 
mcp = FastMCP("Travel Assistant")

@mcp.tool()
def get_weather(city:str)->str:
    """Get Weather information of a city."""


    weather_data = {
        "Goa":"Sunny, 32°C",
        "Hyderabad":"Cloudy, 28°C",
        "Delhi":"Rainy, 15°C"
    }

    return weather_data.get(city,"Weather data unavailable")



@mcp.tool()
def find_attractions(city:str)->str:
    """Find the popular tourist attractions in a city"""

    attractions={
        "Goa":"Baga beach, Fory Aguda, Dudhsagar Falls",
        "Hyderabad":"Charminar, Golkonda Fort, Hussain Sagar",
        "Delhi":"India Gate, Red Fort, Qutub Minar"
            }
    return attractions.get(city,":Attractions unavailable")


if __name__=="__main__":
    mcp.run(transport="stdio")