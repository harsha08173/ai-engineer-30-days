from mcp.server.fastmcp import FastMCP 

mcp = FastMCP("HR Assistant") #Creates MCP Server


employees = {
    "EMP001":{"leave_balance":12,"department":"Engineering"},
    "EMP002":{"leave_balance":7,"department":"HR"},
    "EMP003":{"leave_balance":15,"department":"Finance"}
}


@mcp.tool() #Registers a python function as a mcp tool
def get_leave_balance(employee_id:str):
    """Get an employee's remaining leave balance.""" #describes the tool so AI client ca n understand its purpose
    employee = employees.get(employee_id)

    if not employee:
        return "Employee not found."
    
    return f"Remaining leave: {employee['leave_balance']} days."

@mcp.tool()
def get_employee_department(employee_id:str):
    """Get an employee's department."""

    employee = employees.get(employee_id)
    
    if not employee:
        return "Employee not found."
    
    return employee["department"]


@mcp.resource("hr://leave-policy")
def get_leave_policy() -> str:
    """Return the company leave policy."""
    return """
    Employees receive 20 annual leave days.
    Up to 5 unused days can be carried forward.
    """

@mcp.prompt()
def leave_assistant_prompt() -> str:
    """Provide instructions for an HR leave assistant."""
    return (
        "You are an HR assistant. "
        "Answer questions using the company leave policy. "
        "Be clear and concise."
    )

if __name__ =="__main__":
    mcp.run(transport="stdio") #Starts the server using default transport.


