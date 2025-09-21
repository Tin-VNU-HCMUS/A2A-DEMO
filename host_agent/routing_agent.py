# Fixed routing_agent: explicit intent classification + task_id handling
# Created to replace original routing_agent.py to ensure:
# - Symptom-only queries go only to Symptom Agent
# - Cost-only queries go only to Cost Agent
# - Combined queries call Symptom Agent first, then Cost Agent
# - Do NOT forward a task_id that belongs to another agent (this caused "Task ... was specified but does not exist")

# NOTE: paste this file back as host_agent/routing_agent.py (or replace original) and restart your services.

# Bản sửa lúc 2025-09-19
import asyncio
import base64
import json
import uuid
from typing import Any, Dict, List
from types import SimpleNamespace

import logging
import sys
logger = logging.getLogger("HostAgent")
logger.setLevel(logging.DEBUG)
# add handler if none
if not logger.handlers:
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
from google.adk.sessions.session import Session
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext, new_invocation_context_id
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.agents.base_agent import BaseAgent
from .remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback

# --- Minimal Dummy session/agent utilities (unchanged) ---
class DummySessionService(BaseSessionService):
    def __init__(self, session: Session):
        self._session = session

    async def save_session(self, session: Session) -> None:
        self._session = session

    async def load_session(self, session_id: str) -> Session:
        return self._session

    async def create_session(self, session: Session) -> None:
        self._session = session

    async def delete_session(self, session_id: str) -> None:
        self._session = None

    async def get_session(self, session_id: str) -> Session:
        return self._session

    async def list_sessions(self) -> list[Session]:
        return [self._session] if self._session else []

class DummyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="dummy")

# --- HostAgent ---
class HostAgent:
    """The host agent.

    Responsibilities:
      - route user input to the correct remote agent(s) based on a simple intent classifier
      - avoid passing task ids across agents (a major cause of the "task not found" error)
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
    ):
        # remote state
        self.httpx_client = http_client
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''

        # container to accumulate streaming tasks
        self._tasks: dict[str, SimpleNamespace] = {}

        # callback default
        if task_callback is None:
            self.task_callback = self._default_task_callback
        else:
            self.task_callback = task_callback

        loop = asyncio.get_running_loop()
        # Start background init
        loop.create_task(self.init_remote_agent_addresses(remote_agent_addresses))

        # Intent keyword sets (can be tuned)
        self._symptom_keywords = [
            "triệu chứng",
            "triệu chứng này",
            "bị",
            "nôn",
            "đau",
            "ho",
            "sốt",
            "chảy máu",
            "phân đen",
            "nôn ra máu",
            "nội soi",
            "giãn tĩnh mạch",
            "bệnh gì",
        ]
        self._cost_keywords = [
            "chi phí",
            "giá",
            "gói khám",
            "bao nhiêu",
            "tốn",
            "chi phí khám",
            "giá tiền",
        ]
        self._booking_keywords = [
            "đặt lịch",
            "hẹn khám",
            "booking",
            "đặt khám",
            "muốn đặt",
            "muốn hẹn",
        ]

    async def init_remote_agent_addresses(
        self, remote_agent_addresses: list[str]
    ):
        async with asyncio.TaskGroup() as task_group:
            for address in remote_agent_addresses:
                task_group.create_task(self.retrieve_card(address))

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

        Important fix: DO NOT reuse a task_id that belongs to another agent. We compute
        the outgoing_task_id based on the *previous* agent recorded in the session state.
        """
        logger.info(f"[HostAgent.send_message] Sending to agent: {agent_name}")
        logger.debug(f"[HostAgent.send_message] message preview: {message[:400]}")

        if agent_name not in self.remote_agent_connections:
            logger.error(f"[HostAgent.send_message] Agent not found: {agent_name}")
            raise ValueError(f'Agent {agent_name} not found')

        state = tool_context.state

        # IMPORTANT: compute previous agent BEFORE we overwrite it below.
        previous_agent = state.get('agent')
        # Only forward the stored task_id if it belongs to the same agent we are about to call.
        outgoing_task_id = state.get('task_id') if previous_agent == agent_name else None

        # Now set the active agent in state (this records the intent to talk to this remote)
        state['agent'] = agent_name

        client = self.remote_agent_connections[agent_name]
        if not client:
            logger.error(f"[HostAgent.send_message] Client not available for {agent_name}")
            raise ValueError(f'Client not available for {agent_name}')

        # Keep context id (session-global) if present
        contextId = state.get('context_id', None)

        messageId = state.get('message_id', None)
        if not messageId:
            messageId = str(uuid.uuid4())

        # Build MessageSendParams: pass outgoing_task_id (may be None) so we do NOT accidentally
        # instruct other agents to look up a task_id that doesn't belong to them.
        request: MessageSendParams = MessageSendParams(
            message=Message(
                role='user',
                parts=[TextPart(text=message)],
                messageId=messageId,
                contextId=contextId,
                taskId=outgoing_task_id,
            ),
            configuration=MessageSendConfiguration(
                acceptedOutputModes=['text', 'text/plain', 'image/png'],
            ),
        )

        logger.info(f"[HostAgent.send_message] Sending request messageId={messageId} to {agent_name}")
        logger.debug(f"[HostAgent.send_message] Request preview: {request}")

        # Send and collect response
        response = await client.send_message(request, self.task_callback)

        logger.info(f"[HostAgent.send_message] Received response type: {type(response)} from {agent_name}")
        try:
            logger.debug(f"[HostAgent.send_message] Response (truncated): {str(response)[:2000]}")
        except Exception:
            logger.debug("[HostAgent.send_message] Response could not be stringified for debug")

        # If remote returned a final Message
        if isinstance(response, Message):
            converted = await convert_parts(response.parts, tool_context)
            logger.debug(f"[HostAgent.send_message] Converted message parts: {converted}")
            return converted

        if isinstance(response, (list, str, dict)):
            logger.debug("[HostAgent.send_message] Remote returned immediate list/str/dict - returning as-is")
            return response

        if response is None:
            logger.error("[HostAgent.send_message] Received None response from remote agent - did not collect streaming events")
            raise ValueError("No response from remote agent; ensure HostAgent has a task_callback that accumulates streaming events.")

        if not hasattr(response, "status"):
            logger.error(f"[HostAgent.send_message] Unexpected response object without 'status': {type(response)}")
            raise ValueError(f"Unexpected response type from agent {agent_name}: {type(response)}")

        task = response  # type: ignore

        # Update session state safely. Note: task.id belongs to the remote agent we just called.
        state['session_active'] = task.status.state not in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.unknown,
        ]
        if getattr(task, "contextId", None):
            state['context_id'] = task.contextId
        # store the task id returned by the agent we just called
        state['task_id'] = getattr(task, "id", None)

        if task.status.state == TaskState.input_required:
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.canceled:
            logger.error(f"[HostAgent.send_message] Agent {agent_name} task {getattr(task, 'id', None)} is cancelled")
            raise ValueError(f'Agent {agent_name} task {getattr(task, 'id', None)} is cancelled')
        elif task.status.state == TaskState.failed:
            logger.error(f"[HostAgent.send_message] Agent {agent_name} task {getattr(task, 'id', None)} failed")
            raise ValueError(f'Agent {agent_name} task {getattr(task, 'id', None)} failed')

        response_parts: List[Any] = []
        if getattr(task.status, "message", None):
            response_parts.extend(
                await convert_parts(task.status.message.parts, tool_context)
            )
        if getattr(task, "artifacts", None):
            for artifact in task.artifacts:
                response_parts.extend(
                    await convert_parts(artifact.parts, tool_context)
                )
        logger.debug(f"[HostAgent.send_message] Final response_parts: {response_parts}")
        return response_parts

    def _default_task_callback(self, event, card: AgentCard):
        """
        Accumulate streaming events into a Task-like object.
        """
        if isinstance(event, Task) or isinstance(event, Message):
            return event

        # task id extraction (defensive)
        task_id = None
        if hasattr(event, "taskId"):
            task_id = getattr(event, "taskId")
        elif hasattr(event, "task_id"):
            task_id = getattr(event, "task_id")
        elif hasattr(event, "id"):
            task_id = getattr(event, "id")
        elif isinstance(event, dict):
            task_id = event.get("taskId") or event.get("task_id") or event.get("id")

        if not task_id:
            task_id = str(uuid.uuid4())

        t = self._tasks.get(task_id)
        if not t:
            t = SimpleNamespace(
                id=task_id,
                contextId=None,
                artifacts=[],
                status=SimpleNamespace(state=TaskState.working, message=None),
            )
            self._tasks[task_id] = t

        # update contextId
        if hasattr(event, "contextId"):
            t.contextId = getattr(event, "contextId")
        elif isinstance(event, dict) and "contextId" in event:
            t.contextId = event.get("contextId")

        # update status
        status = None
        if hasattr(event, "status"):
            status = getattr(event, "status")
        elif isinstance(event, dict) and "status" in event:
            status = event.get("status")

        if status:
            if hasattr(status, "state"):
                t.status.state = getattr(status, "state")
            elif isinstance(status, dict) and "state" in status:
                t.status.state = status.get("state", t.status.state)

            message = getattr(status, "message", None) if hasattr(status, "message") else (status.get("message") if isinstance(status, dict) else None)
            if message:
                t.status.message = message

        # artifact
        artifact = None
        if hasattr(event, "artifact"):
            artifact = getattr(event, "artifact")
        elif isinstance(event, dict) and "artifact" in event:
            artifact = event.get("artifact")

        if artifact:
            # NOTE: we keep appending; duplication might occur upstream. We can dedupe later.
            t.artifacts.append(artifact)

        return t

    async def orchestrate_care_flow(
        self, message: str, tool_context: ToolContext, wants_symptom: bool = True, wants_cost: bool = True, wants_booking: bool = False
    ):
        """
        Orchestrate the triage & booking flow with explicit flags.

        Returns a dict with keys: symptom, cost, booking (if executed).
        """
        results: Dict[str, Any] = {}

        summary_for_cost = ""

        # Step 1: SymptomAgent (only if requested)
        if wants_symptom and "Symptom Agent" in self.remote_agent_connections:
            try:
                logger.info("=== Gọi Symptom Agent ===")
                symptom_resp = await self.send_message("Symptom Agent", message, tool_context)
                logger.info(f"[Symptom Agent] Kết quả: {symptom_resp}")
            except Exception as e:
                symptom_resp = [f"Error calling Symptom Agent: {e}"]
                logger.error(symptom_resp)
        elif wants_symptom:
            symptom_resp = ["Symptom Agent not available"]
            logger.warning("Symptom Agent not available")
        else:
            symptom_resp = ["Not requested"]

        results["symptom"] = symptom_resp

        # Prepare summary if Symptom Agent ran
        if wants_symptom:
            summary_for_cost = self._summarize_symptom_result(symptom_resp)

        # Step 2: CostAgent (only if requested)
        if wants_cost and "Cost Agent" in self.remote_agent_connections:
            try:
                logger.info("=== Gọi CostAgent ===")
                # If we have a good summary, use it; else use the raw user message as prompt
                cost_prompt = (
                    f"Tóm tắt từ Symptom Agent: {summary_for_cost}\n\n"
                    f"User hỏi: {message}\n\n"
                    "Hãy đề xuất các gói khám, xét nghiệm, và chi phí tương ứng."
                )
                cost_resp = await self.send_message("Cost Agent", cost_prompt if summary_for_cost else message, tool_context)
                logger.info(f"[Cost Agent] Kết quả: {cost_resp}")
            except Exception as e:
                cost_resp = [f"Error calling Cost Agent: {e}"]
                logger.error(cost_resp)
        elif wants_cost:
            cost_resp = ["Cost Agent not available"]
            logger.warning("Cost Agent not available")
        else:
            cost_resp = ["Not requested"]

        results["cost"] = cost_resp

        # Step 3: BookingAgent (unchanged logic)
        if wants_booking:
            wants_booking_flag = any(kw in message.lower() for kw in self._booking_keywords)
        else:
            wants_booking_flag = False

        if wants_booking_flag and "BookingAgent" in self.remote_agent_connections:
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
        if symptom_resp is None:
            return ""
        if isinstance(symptom_resp, str):
            return symptom_resp.strip()
        if isinstance(symptom_resp, list):
            pieces = []
            for item in symptom_resp:
                if isinstance(item, dict):
                    pieces.append(item.get("text") or json.dumps(item))
                else:
                    pieces.append(str(item))
                if len(pieces) >= 6:
                    break
            summary = " | ".join(pieces)
            return summary[:150] + ("..." if len(summary) > 150 else "")
        return str(symptom_resp)[:150]

    # === Thêm method ainvoke để HostAgent dùng trực tiếp làm agent ===
    async def ainvoke(self, message: str, session_id: str = None, **kwargs):
        logger.info(f"[HostAgent.ainvoke] Processing message: {message[:200]}")

        # Tạo session đầy đủ
        session = Session(
            id=session_id or "demo-session",
            app_name="a2a-hospital-agents",
            user_id="demo-user",
            state={}
        )

        session_service = DummySessionService(session)

        invocation_context = InvocationContext(
            session_service=session_service,
            invocation_id=new_invocation_context_id(),
            agent=DummyAgent(),
            session=session
        )

        tool_context = ToolContext(invocation_context)

        # --- intent classification (base) ---
        msg_lower = (message or "").lower()
        wants_symptom = any(kw in msg_lower for kw in self._symptom_keywords)
        wants_cost = any(kw in msg_lower for kw in self._cost_keywords)
        wants_booking = any(kw in msg_lower for kw in self._booking_keywords)

        # --- heuristics to resolve ambiguous queries where both symptom+cost keywords appear ---
        # If the user's wording clearly requests packages/prices ("gói khám", "cho tôi các gói", "chi phí", "bao nhiêu"),
        # prefer routing to Cost Agent alone unless the user explicitly asks about symptoms ("triệu chứng", "triệu chứng gì", "dấu hiệu").
        cost_priority_phrases = [
            "gói khám", "các gói khám", "chi phí", "giá", "bao nhiêu", "tốn", "chi phí khám",
            "cho tôi các gói", "cho tôi gói", "xin cho biết", "cho biết", "tư vấn gói", "gợi ý gói",
        ]
        symptom_question_phrases = [
            "triệu chứng gì", "triệu chứng", "dấu hiệu", "bị ", "bị", "nôn", "đau", "sốt", "chảy máu",
            "phân đen", "nôn ra máu", "nội soi", "bệnh gì",
        ]

        cost_cue = any(p in msg_lower for p in cost_priority_phrases)
        symptom_cue = any(p in msg_lower for p in symptom_question_phrases)

        # Resolve precedence:
        # - If cost cue is present and no explicit symptom question cue -> treat as cost-only.
        # - If symptom cue is present and no cost cue -> treat as symptom-only.
        # - Otherwise (both present or neither) keep both flags as-is and orchestrate both (fallback to symptom first).
        if wants_cost and cost_cue and not symptom_cue:
            wants_symptom = False
            wants_cost = True
        elif wants_symptom and symptom_cue and not cost_cue:
            wants_cost = False

        logger.debug(f"[HostAgent.ainvoke] intent wants_symptom={wants_symptom}, wants_cost={wants_cost}, wants_booking={wants_booking} (after heuristics cost_cue={cost_cue}, symptom_cue={symptom_cue})")

        # Route according to explicit intent rules required by user:
        # - symptom-only -> Symptom Agent
        # - cost-only -> Cost Agent
        # - both -> Symptom Agent then Cost Agent (use orchestrate_care_flow)
        if wants_symptom and wants_cost:
            return await self.orchestrate_care_flow(message, tool_context, wants_symptom=True, wants_cost=True, wants_booking=wants_booking)
        elif wants_symptom and not wants_cost:
            return await self.send_message("Symptom Agent", message, tool_context)
        elif wants_cost and not wants_symptom:
            return await self.send_message("Cost Agent", message, tool_context)
        else:
            # fallback: route to Symptom Agent (more likely to be medical question)
            logger.debug("[HostAgent.ainvoke] fallback: routing to Symptom Agent")
            return await self.send_message("Symptom Agent", message, tool_context)


async def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(await convert_part(p, tool_context))
    return rval


async def convert_part(part: Part, tool_context: ToolContext):
    try:
        root = part.root
    except Exception:
        return str(part)

    kind = getattr(root, "kind", None)
    if kind == 'text':
        return getattr(root, "text", "")
    if kind == 'data':
        return getattr(root, "data", {})
    if kind == 'file':
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


# ===== Helper khởi tạo đồng bộ =====n
def get_initialized_routing_agent_sync(remote_agent_addresses: list[str]):
    import asyncio

    async def _init():
        client = httpx.AsyncClient(timeout=30)
        host = HostAgent(remote_agent_addresses, client)
        # Đợi load xong danh thiếp từ các remote agent
        await host.init_remote_agent_addresses(remote_agent_addresses)
        print(">>> DEBUG: returning HostAgent, not LlmAgent", type(host))
        return host

    return asyncio.get_event_loop().run_until_complete(_init())
