import asyncio
import uvicorn
from typing import Any

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agents.booking_agent.agent_executor import BookingAgentExecutor
from agents.booking_agent.booking_agent import BookingAgent

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    USE_MCP = True
except ImportError:
    USE_MCP = False

SERVER_CONFIGS = {
    "toolset_server_urls": ["http://localhost:9003"]
}

app_context: dict[str, Any] = {}

async def app_lifespan(context: dict[str, Any]):
    if USE_MCP:
        mcp_client = MultiServerMCPClient(SERVER_CONFIGS)
        tools = await mcp_client.get_tools()
        context["mcp_tools"] = tools
    else:
        context["mcp_tools"] = []

    yield

    if USE_MCP:
        await mcp_client.__aexit__(None, None, None)
    context.clear()

def get_agent_card(host: str, port: int) -> AgentCard:
    return AgentCard(
        name="Booking Agent",
        description="Agent đặt lịch khám bệnh",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=BookingAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=BookingAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(streaming=True, pushNotifications=True),
        skills=[
            AgentSkill(
                id="booking_appointment",
                name="Đặt lịch khám",
                description="Đặt lịch hẹn bác sĩ hoặc khám bệnh",
                tags=["lịch khám", "đặt chỗ"],
                examples=["Tôi muốn đặt lịch khám vào thứ 2", "Đặt lịch bác sĩ nội tổng quát"]
            )
        ]
    )

async def run_server_async(host: str, port: int, log_level: str):
    async with app_lifespan(app_context):
        mcp_tools = app_context.get("mcp_tools", [])
        agent_executor = BookingAgentExecutor(mcp_tools)

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore()
        )

        app = A2AStarletteApplication(
            agent_card=get_agent_card(host, port),
            request_handler=request_handler,
            task_store=InMemoryTaskStore()
        )

        config = uvicorn.Config(app=app.build(), host=host, port=port, log_level=log_level, lifespan="on")
        await uvicorn.Server(config).serve()

def main(host="localhost", port=10003, log_level="info"):
    asyncio.run(run_server_async(host, port, log_level))

if __name__ == "__main__":
    main()
