"""Custom MCP server — Python + FastAPI equivalent of custom-mcp-node.

Exposes tools and resource:
  - create_support_ticket: Create a new IT support ticket.
  - check_support_status: Retrieve the status of an existing ticket

Run with: uvicorn server:app --port 3000
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from mcp.server import MCPServer

mcp_server = MCPServer(name="IT_HelpDesk_MCP_Server", version="1.0.0")

SEED_FILE = Path(__file__).parent / "mcp_tool_data.json"


def _load_seed(path: Path) -> dict[str, dict]:
    """Load seed tickets from JSON into an in-memory dict keyed by ticket_id."""
    with open(path, "r", encoding="utf-8") as f:
        return {t["ticket_id"]: t for t in json.load(f)}


# In-memory ticket store: {ticket_id: ticket}. Resets to the seed on restart.
tickets: dict[str, dict] = _load_seed(SEED_FILE)


def _next_ticket_id() -> str:
    numbers = [int(tid.split("-")[1]) for tid in tickets if tid.startswith("IT-")]
    return f"IT-{max(numbers, default=100) + 1}"


@mcp_server.tool(name="create_support_ticket", description="Create a new IT support ticket")
def create_support_ticket(employee: str, issue: str, description: str, priority: str) -> str:
    ticket = {
        "ticket_id": _next_ticket_id(),
        "employee": employee,
        "issue": issue,
        "description": description,
        "priority": priority,
        "status": "Open",
    }
    tickets[ticket["ticket_id"]] = ticket
    return f"Ticket successfully created: {ticket}"


@mcp_server.tool(name="check_support_status", description="Retrieve the status of an existing ticket")
def check_support_status(ticket_id: str) -> str:
    ticket = tickets.get(ticket_id.strip().upper())
    if ticket is None:
        return f"Ticket {ticket_id} not found"
    return f"Ticket {ticket['ticket_id']} status: {ticket['status']} ({ticket['issue']}, priority {ticket['priority']})"


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
