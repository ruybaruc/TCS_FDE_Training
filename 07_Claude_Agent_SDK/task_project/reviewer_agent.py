import asyncio
import os
from claude_agent_sdk import (
    query, ClaudeAgentOptions, AssistantMessage, ResultMessage, ToolUseBlock,
)
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "claude-haiku-5-5"
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_project")


async def _scout(prompt):
    options = ClaudeAgentOptions(
        model=MODEL_NAME,
        cwd=PROJECT_DIR,
        tools=["Glob", "Grep", "Read"],
        allowed_tools=["Glob", "Grep", "Read"],
        disallowed_tools=["Bash"],
    )
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"[tool] {block.name}")
        elif isinstance(message, ResultMessage):
            print(message.result)
            print(f"Turns: {message.num_turns}")
            print(f"Cost: ${message.total_cost_usd}")


def scout(prompt):
    """Run a read-only review agent. Print each tool it uses, the final report, turns and cost."""
    asyncio.run(_scout(prompt))


if __name__ == "__main__":
    scout("List every function in the project and say which ones have no docstring.")
    scout("Report any bugs or security problems in the project, each with a one-line fix.")