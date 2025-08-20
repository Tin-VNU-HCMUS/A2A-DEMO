from typing import Literal, Any, AsyncIterable
import asyncio
import uvicorn

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agents.symptom_agent.agent_executor import SymptomAgentExecutor
from agents.symptom_agent.symptom_agent import SymptomAgent

from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging, os, httpx
import sys


# Nếu bạn đang dùng MCP Toolset:
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    USE_MCP = True
except ImportError:
    USE_MCP = False


load_dotenv(override=True)

SERVER_CONFIGS = {
    'bnb': {
        'command': 'npx',
        'args': ['-y', '@openbnb/mcp-server-airbnb', '--ignore-robots-txt'],
        'transport': 'stdio',
    },
}

app_context: dict[str, Any] = {}


DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 10002
DEFAULT_LOG_LEVEL = 'info'



app_context: dict[str, Any] = {}

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 10002
DEFAULT_LOG_LEVEL = 'info'

# =====================
# 2. MCP client lifespan
# =====================
@asynccontextmanager
async def app_lifespan(context: dict[str, Any]):
    print('Lifespan: Initializing MCP client and tools...')

    mcp_client_instance: MultiServerMCPClient | None = None

    try:
        if MultiServerMCPClient is not None:
            mcp_client_instance = MultiServerMCPClient(SERVER_CONFIGS)
            mcp_tools = await mcp_client_instance.get_tools()
            context['mcp_tools'] = mcp_tools
            print(f'Lifespan: MCP Tools preloaded successfully ({len(mcp_tools)} tools found).')
        else:
            print("Warning: langchain_mcp_adapters not installed. Skipping MCP tools.")
            context['mcp_tools'] = []

        yield
    except Exception as e:
        print(f'Lifespan: Error during initialization: {e}', file=sys.stderr)
        raise
    finally:
        print('Lifespan: Shutting down MCP client...')
        if mcp_client_instance and hasattr(mcp_client_instance, '__aexit__'):
            try:
                await mcp_client_instance.__aexit__(None, None, None)
                print('Lifespan: MCP Client resources released via __aexit__.')
            except Exception as e:
                print(f'Lifespan: Error during MCP client __aexit__: {e}', file=sys.stderr)

        print('Lifespan: Clearing application context.')
        context.clear()

# =====================
# 3. Agent Card setup
# =====================
def get_agent_card(host: str, port: int) -> AgentCard:
    return AgentCard(
        name='Symptom Agent',
        description='Agent tư vấn triệu chứng y tế ban đầu',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=SymptomAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=SymptomAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(streaming=True, pushNotifications=True),
        skills=[
            AgentSkill(
                id='symptom_search',
                name='Search Symptom',
                description='Tìm kiếm và xử lý thông tin triệu chứng',
                tags=['triệu chứng', 'bệnh học'],
                examples=['Tôi bị ho và sốt nhẹ', 'Tôi đau bụng vùng dưới']
            )
        ]
    )

# =====================
# 4. MAIN (CLI)
# =====================
def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv('GOOGLE_API_KEY'):
        raise ValueError(
            'GOOGLE_API_KEY environment variable not set and '
            'GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
        )

    async def run_server_async():
        async with app_lifespan(app_context):
            mcp_tools = app_context.get('mcp_tools', [])
            agent_executor = SymptomAgentExecutor(mcp_tools=mcp_tools)

            request_handler = DefaultRequestHandler(
                agent_executor=agent_executor,
                task_store=InMemoryTaskStore(),
            )

            a2a_server = A2AStarletteApplication(
                agent_card=get_agent_card(host, port),
                http_handler=request_handler,
            )

            asgi_app = a2a_server.build()

            config = uvicorn.Config(
                app=asgi_app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                lifespan='auto',
            )

            print(f'Starting Uvicorn server at http://{host}:{port} with log-level {log_level}...')
            try:
                await uvicorn.Server(config).serve()
            except KeyboardInterrupt:
                print('Server shutdown requested (KeyboardInterrupt).')
            finally:
                print('Uvicorn server has stopped.')

    try:
        asyncio.run(run_server_async())
    except RuntimeError as e:
        print(f'RuntimeError: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Unexpected Error: {e}', file=sys.stderr)
        sys.exit(1)

# =====================
# 5. Entry point
# =====================
if __name__ == '__main__':
    main()