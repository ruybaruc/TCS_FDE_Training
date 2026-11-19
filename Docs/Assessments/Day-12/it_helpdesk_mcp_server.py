"""Custom MCP server — Python + FastAPI equivalent of custom-mcp-node.

Exposes tools and resource:
  - create_support_ticket: Create a new IT support ticket.
  - check_support_status: Retrieve the status of an existing ticket

Run with: uvicorn server:app --port 3000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp.server import MCPServer

mcp_server = MCPServer(name="IT_HelpDesk_MCP_Server", version="1.0.0")

@mcp_server.tool(name="create_support_ticket", description="Create a new IT support ticket")
def create_support_ticket(employee: str, issue: str, description: str, priority: str) -> str:
    ticket = {
        "ticket_id": "IT-101",
        "employee": "Rahul",
        "issue": "Laptop running slowly",
        "description": "System takes too long to start",
        "priority": "Medium",
        "status": "Open"
    }
    return f"Ticket successfully created: {ticket}"

@mcp_server.tool(name="check_support_status", description="Retrieve the status of an existing ticket")
def check_support_status(ticket_id: str) -> str:
    return "Ticket status: In Progress"


mcp_app = mcp_server.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # mcp_app's own lifespan starts the MCP session manager's task group.
    # FastAPI doesn't run a mounted sub-app's lifespan automatically, so we
    # enter it explicitly here to keep that startup/shutdown tied to ours.
    async with mcp_app.router.lifespan_context(mcp_app):
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/", mcp_app)

if __name__ == "__main__":
    import asyncio
    import sys

    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=3000)
    server = uvicorn.Server(config)

    # On Windows, uvicorn's default event loop is a ProactorEventLoop, which
    # logs a noisy (but harmless) ConnectionResetError whenever a client
    # disconnects abruptly — which our short-lived-connection-per-request
    # client does on every call. Running under a SelectorEventLoop instead
    # avoids that; asyncio.run(loop_factory=...) is the non-deprecated way
    # to choose the loop type (the older set_event_loop_policy is removed
    # in newer Python versions).
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None

    print("MCP server running at http://localhost:3000")
    asyncio.run(server.serve(), loop_factory=loop_factory)
