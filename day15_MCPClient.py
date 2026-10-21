import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command = "python",
        args = ["day15_MCPServer.py"]
    )

    async with stdio_client(server_params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()

            
            prompt = await session.get_prompt("leave_assistant_prompt")
            print("Prompt: ",prompt.messages)

            resource = await session.read_resource("hr://leave-policy")
            print("Leave Policy:", resource.contents)

            tools = await session.list_tools()
            print("Available tools: ")
            for tool in tools.tools:
                print(tool.name)

            result = await session.call_tool(
                "get_leave_balance",
                arguments = {"employee_id":"EMP002"}
            )

            print("Result: ",result.content)

asyncio.run(main())