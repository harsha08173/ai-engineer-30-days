import asyncio 
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    
    server_params = StdioServerParameters(
        command = "python",
        args = ["day16_Travel_MCPServer.py"]
    )

    async with stdio_client(server_params) as (read,write):

        async with ClientSession(read,write) as session:

            await session.initialize()
            tools = await session.list_tools()

            for tool in tools.tools:
                print(tool.name)

            weather = await session.call_tool(
                "get_weather", arguments = {"city":"Goa"})

            print("\nGoa Weather:",weather.content[0].text)

            places = await session.call_tool(
                "find_attractions",
                arguments = {"city":"Goa"}
            )            

            print("\nGoa attractions:",places.content[0].text)


if __name__ =="__main__":
    asyncio.run(main()) 