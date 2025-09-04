import asyncio
import base64
import json
import uuid
from typing import Any, Dict, List


import logging
logger = logging.getLogger("HostAgent")
logger.setLevel(logging.DEBUG)
# add handler if none
if not logger.handlers:
    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(ch)


import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Task,
    TaskState,
    TextPart,
)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback


class HostAgent:
    """The host agent.

    This agent orchestrates a 3-step care flow:
      1) send symptom text to SymptomAgent -> get differential diagnosis + tests
      2) send summary to CostAgent -> get packages & prices
      3) optionally send booking request to BookingAgent
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.httpx_client = http_client
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''
        loop = asyncio.get_running_loop()
        # Start background init (it's still possible to wait synchronously later).
        loop.create_task(self.init_remote_agent_addresses(remote_agent_addresses))

    async def init_remote_agent_addresses(
        self, remote_agent_addresses: list[str]
    ):
        async with asyncio.TaskGroup() as task_group:
            for address in remote_agent_addresses:
                task_group.create_task(self.retrieve_card(address))
        # Once completed, self.agents is populated by register_agent_card.

    async def retrieve_card(self, address: str):
        card_resolver = A2ACardResolver(self.httpx_client, address)
        card = await card_resolver.get_agent_card()
        self.register_agent_card(card)

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(self.httpx_client, card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        return Agent(
            model='gemini-2.0-flash-001',
            name='host_agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                'This agent orchestrates the decomposition of the user request into'
                ' tasks that can be performed by the child agents.'
            ),
            tools=[
                self.list_remote_agents,
                self.send_message,
                self.orchestrate_care_flow,  # orchestration tool exposed to LLM
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Clear instruction so model will prefer orchestration tool for combined requests."""
        current_agent = self.check_state(context)
        return f"""
Bạn là điều phối viên. KHÔNG trả lời trực tiếp khi có thể sử dụng tools.
Nếu người dùng mô tả TRIỆU CHỨNG và/hoặc hỏi về GÓI KHÁM / CHI PHÍ / ĐẶT LỊCH:
  - Sử dụng tool `orchestrate_care_flow(message)` để chạy luồng:
      1) Gọi SymptomAgent phân tích triệu chứng và đưa ra các chẩn đoán gợi ý + danh sách xét nghiệm cần làm.
      2) Gọi CostAgent với tóm tắt từ SymptomAgent để lấy các gói khám & chi phí đề xuất.
      3) Nếu người dùng rõ ràng muốn ĐẶT LỊCH, gọi BookingAgent để đặt lịch.
  - Nếu chỉ cần 1 bước đơn giản, có thể dùng `send_message(agent_name, message)` trực tiếp.

Luôn bắt đầu bằng `list_remote_agents()` nếu cần kiểm tra agent hiện có.
Không bịa thông tin. Trả lời cuối cùng phải tổng hợp kết quả từ các agent đã gọi.

Agents hiện có:
{self.agents}

Current agent: {current_agent['active_agent']}
        """

    def check_state(self, context: ReadonlyContext):
        state = context.state
        if (
            'context_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'agent' in state
        ):
            return {'active_agent': f'{state["agent"]}'}
        return {'active_agent': 'None'}

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request
    ):
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            state['session_active'] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_message(
        self, agent_name: str, message: str, tool_context: ToolContext
    ):
        """Sends a message to a remote agent and returns a normalized response.

        Returns:
          - If remote agent returns a Message or Parts -> returns list of strings / dicts
          - If it returns a Task (async), waits for final status and extracts parts/artifacts.
        """
        logger.info(f"[HostAgent.send_message] Sending to agent: {agent_name}")
        logger.debug(f"[HostAgent.send_message] message preview: {message[:400]}")

        if agent_name not in self.remote_agent_connections:
            logger.error(f"[HostAgent.send_message] Agent not found: {agent_name}")
            raise ValueError(f'Agent {agent_name} not found')

        state = tool_context.state
        state['agent'] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            logger.error(f"[HostAgent.send_message] Client not available for {agent_name}")
            raise ValueError(f'Client not available for {agent_name}')

        taskId = state.get('task_id', None)
        contextId = state.get('context_id', None)
        messageId = state.get('message_id', None)
        if not messageId:
            messageId = str(uuid.uuid4())

        request: MessageSendParams = MessageSendParams(
            id=str(uuid.uuid4()),
            message=Message(
                role='user',
                parts=[TextPart(text=message)],
                messageId=messageId,
                contextId=contextId,
                taskId=taskId,
            ),
            configuration=MessageSendConfiguration(
                acceptedOutputModes=['text', 'text/plain', 'image/png'],
            ),
        )

        logger.info(f"[HostAgent.send_message] Sending request id={request.id} to {agent_name}")
        logger.debug(f"[HostAgent.send_message] Request preview: {str(request)[:800]}")

        response = await client.send_message(request, self.task_callback)

        logger.info(f"[HostAgent.send_message] Received response type: {type(response)} from {agent_name}")
        try:
            logger.debug(f"[HostAgent.send_message] Response (truncated): {str(response)[:2000]}")
        except Exception:
            # some response objects may not stringify nicely
            logger.debug("[HostAgent.send_message] Response could not be stringified for debug")

        # If remote returns immediate Message -> convert parts and return
        if isinstance(response, Message):
            converted = await convert_parts(response.parts, tool_context)
            logger.debug(f"[HostAgent.send_message] Converted message parts: {converted}")
            return converted

        # Otherwise response is Task: update session state and collect final outputs
        task: Task = response
        state['session_active'] = task.status.state not in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.unknown,
        ]
        if task.contextId:
            state['context_id'] = task.contextId
        state['task_id'] = task.id

        if task.status.state == TaskState.input_required:
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.canceled:
            logger.error(f"[HostAgent.send_message] Agent {agent_name} task {task.id} is cancelled")
            raise ValueError(f'Agent {agent_name} task {task.id} is cancelled')
        elif task.status.state == TaskState.failed:
            logger.error(f"[HostAgent.send_message] Agent {agent_name} task {task.id} failed")
            raise ValueError(f'Agent {agent_name} task {task.id} failed')

        response_parts: List[Any] = []
        if task.status.message:
            response_parts.extend(
                await convert_parts(task.status.message.parts, tool_context)
            )
        if task.artifacts:
            for artifact in task.artifacts:
                response_parts.extend(
                    await convert_parts(artifact.parts, tool_context)
                )
        logger.debug(f"[HostAgent.send_message] Final response_parts: {response_parts}")
        return response_parts


    import logging
    logger = logging.getLogger("HostAgent")

    async def orchestrate_care_flow(
        self, message: str, tool_context: ToolContext
    ):
        """
        Orchestrate the triage & booking flow:
        1) Send symptom text to SymptomAgent -> get diagnosis + tests
        2) Summarize and send to CostAgent -> get packages & prices
        3) If user intent contains booking keywords, call BookingAgent
        Returns a dict with keys: symptom, cost, booking (if executed).
        """
        results: Dict[str, Any] = {}

        # Step 1: SymptomAgent
        if "SymptomAgent" in self.remote_agent_connections:
            try:
                logger.info("=== Gọi SymptomAgent ===")
                symptom_resp = await self.send_message(
                    "SymptomAgent", message, tool_context
                )
                logger.info(f"[SymptomAgent] Kết quả: {symptom_resp}")
            except Exception as e:
                symptom_resp = [f"Error calling SymptomAgent: {e}"]
                logger.error(symptom_resp)
        else:
            symptom_resp = ["SymptomAgent not available"]
            logger.warning("SymptomAgent not available")
        results["symptom"] = symptom_resp

        # Build a concise summary for cost agent
        summary_for_cost = self._summarize_symptom_result(symptom_resp)

        # Step 2: CostAgent
        cost_trigger_keywords = ["giá", "chi phí", "bao nhiêu", "gói khám"]
        wants_cost = any(kw in message.lower() for kw in cost_trigger_keywords)

        if "CostAgent" in self.remote_agent_connections and (summary_for_cost or wants_cost):
            try:
                logger.info("=== Gọi CostAgent ===")
                cost_prompt = (
                    f"Tóm tắt từ SymptomAgent: {summary_for_cost}\n\n"
                    f"User hỏi: {message}\n\n"
                    "Hãy đề xuất các gói khám, xét nghiệm, và chi phí tương ứng."
                )
                cost_resp = await self.send_message("CostAgent", cost_prompt, tool_context)
                logger.info(f"[CostAgent] Kết quả: {cost_resp}")
            except Exception as e:
                cost_resp = [f"Error calling CostAgent: {e}"]
                logger.error(cost_resp)
        else:
            cost_resp = ["CostAgent not available or no summary"]
            logger.warning("CostAgent not available or no summary")
        results["cost"] = cost_resp

        # Step 3: BookingAgent
        booking_trigger_keywords = [
            "đặt lịch", "hẹn khám", "booking", "đặt khám", "muốn đặt", "muốn hẹn",
        ]
        wants_booking = any(kw in message.lower() for kw in booking_trigger_keywords)

        if wants_booking and "BookingAgent" in self.remote_agent_connections:
            try:
                logger.info("=== Gọi BookingAgent ===")
                booking_prompt = (
                    "User wants to book an appointment. Use the following info to book:\n"
                    f"Summary: {summary_for_cost}\n"
                    f"Preferred details from user message: {message}\n"
                    "Provide booking confirmation, available slots, or required follow-ups."
                )
                booking_resp = await self.send_message("BookingAgent", booking_prompt, tool_context)
                logger.info(f"[BookingAgent] Kết quả: {booking_resp}")
            except Exception as e:
                booking_resp = [f"Error calling BookingAgent: {e}"]
                logger.error(booking_resp)
            results["booking"] = booking_resp
        else:
            results["booking"] = ["Not requested or BookingAgent not available"]
            logger.info("Không gọi BookingAgent")

        return results

    def _summarize_symptom_result(self, symptom_resp: Any) -> str:
        """Create a compact single-line summary from the symptom agent response."""
        # symptom_resp may be a list of strings/dicts or a single string.
        if symptom_resp is None:
            return ""
        if isinstance(symptom_resp, str):
            return symptom_resp.strip()
        if isinstance(symptom_resp, list):
            # join a few items, truncate long outputs
            pieces = []
            for item in symptom_resp:
                if isinstance(item, dict):
                    pieces.append(item.get("text") or json.dumps(item))
                else:
                    pieces.append(str(item))
                if len(pieces) >= 6:
                    break
            summary = " | ".join(pieces)
            # truncate to reasonable length
            return summary[:150] + ("..." if len(summary) > 150 else "")
        # fallback
        return str(symptom_resp)[:150]



async def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(await convert_part(p, tool_context))
    return rval


async def convert_part(part: Part, tool_context: ToolContext):
    # Note: part.root.kind is expected to be 'text' | 'data' | 'file'
    # We keep the logic defensive in case of unexpected shapes.
    try:
        root = part.root
    except Exception:
        # fallback printing entire part
        return str(part)

    kind = getattr(root, "kind", None)
    if kind == 'text':
        return getattr(root, "text", "")
    if kind == 'data':
        return getattr(root, "data", {})
    if kind == 'file':
        # Repackage A2A FilePart to google.genai Blob
        file_id = getattr(root.file, "name", str(uuid.uuid4()))
        file_bytes_b64 = getattr(root.file, "bytes", None)
        if file_bytes_b64:
            file_bytes = base64.b64decode(file_bytes_b64)
            file_part = types.Part(
                inline_data=types.Blob(
                    mime_type=getattr(root.file, "mimeType", "application/octet-stream"),
                    data=file_bytes,
                )
            )
            await tool_context.save_artifact(file_id, file_part)
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
            return DataPart(data={'artifact-file-id': file_id})
        else:
            return {'file': 'empty'}
    return f'Unknown type: {getattr(part, "kind", str(part))}'

# ===== Helper khởi tạo đồng bộ =====

import httpx

def get_initialized_routing_agent_sync(remote_agent_addresses: list[str]):
    """
    Hàm helper để khởi tạo HostAgent đồng bộ, đảm bảo load xong danh thiếp từ các remote agents
    trước khi trả về Agent (tránh race condition).
    """
    import asyncio

    async def _init():
        client = httpx.AsyncClient(timeout=30)
        host = HostAgent(remote_agent_addresses, client)
        # Đợi load xong danh thiếp từ các remote agent
        await host.init_remote_agent_addresses(remote_agent_addresses)
        return host.create_agent()

    # Chạy async trong vòng lặp hiện tại (blocking)
    return asyncio.get_event_loop().run_until_complete(_init())
