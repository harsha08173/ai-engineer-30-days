
import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def main():
    # Configure how to start the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["day16_Travel_MCPServer.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools provided by the MCP server
            mcp_tools = await session.list_tools()

            # Convert MCP tools into OpenAI tool definitions
            tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "parameters": tool.inputSchema
                }
                for tool in mcp_tools.tools
            ]

            # Ask the AI which tools it needs
            response = openai_client.responses.create(
                model="gpt-5-mini",
                input=(
                    """Plan a short trip to Goa. "
                    Check the weather and suggest tourist attractions."""
                ),
                tools=tools
            )

            # Store the AI's tool calls and their results
            tool_outputs = []

            for item in response.output:
                if item.type == "function_call":
                    print("AI selected tool:", item.name)


                    arguments = json.loads(item.arguments)

                    # Execute the selected tool on the MCP server
                    result = await session.call_tool(
                        item.name,
                        arguments=arguments
                    )

                    # Save the result to send back to the AI
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result.content[0].text
                    })

            # Send the tool results back so the AI can prepare its answer
            if tool_outputs:
                final_response = openai_client.responses.create(
                    model="gpt-5-mini",
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=tools
                )

                print("\nFinal travel plan:")
                print(final_response.output_text)



if __name__ == "__main__":
    asyncio.run(main())
