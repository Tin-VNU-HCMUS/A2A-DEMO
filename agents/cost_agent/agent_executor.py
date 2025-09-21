from typing import Any
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, Part
from a2a.utils import new_agent_text_message, new_text_artifact

from cost_agent import CostAgent
from langchain_core.messages import HumanMessage


# --- Logger setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class CostAgentExecutor(AgentExecutor):
    def __init__(self, mcp_tools: list[Any]):
        self.agent = CostAgent(mcp_tools=mcp_tools)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Nếu chưa có task, đánh dấu là submitted
        if not context.current_task:
            await updater.update_status(TaskState.submitted)

        # Task bắt đầu xử lý
        await updater.update_status(TaskState.working)

        try:
            # --- Wrap input ---
            user_query = context.get_user_input()

            response_parts = []
            if context.current_task:
                response_parts = context.current_task.get("final_response_parts", [])

            wrapped_input = {
                "session_id": context.context_id,
                "query": user_query,
                "final_response_parts": response_parts,
            }

            logger.debug(f"[CostAgentExecutor] wrapped_input: {wrapped_input}")

            # Convert sang HumanMessage
            message = HumanMessage(
                content=user_query,
                additional_kwargs={
                    "session_id": context.context_id,
                    "final_response_parts": response_parts,
                },
            )

            logger.debug(f"[CostAgentExecutor] HumanMessage: {message}")

            # --- Run agent ---
            last_event = None
            async for event in self.agent.stream(message):
                logger.debug(f"[CostAgentExecutor] raw event from agent: {event}")
                last_event = event
                await updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(event["content"]),
                )

            # Nếu có kết quả cuối cùng thì gửi artifact
            if last_event and "content" in last_event:
                # Lấy content và data từ last_event
                content = last_event["content"]
                data = last_event.get("data") if isinstance(last_event, dict) else None

                # Tạo artifact
                artifact = new_text_artifact(
                    name=f"agent_response_{context.task_id}.txt",
                    text=content,
                    description="Final response from CostAgent"
                )
                logger.debug(f"[CostAgentExecutor] Final artifact parts dump: {artifact.parts}")
                await updater.add_artifact(artifact.parts)  # Truyền parts thay vì [artifact]

            else:
                logger.warning("No valid last_event or content missing, skipping artifact creation")

            await updater.update_status(TaskState.completed, final=True)

        except Exception as e:
            logger.error(f"CostAgentExecutor error: {e}", exc_info=True)
            await updater.update_status(TaskState.failed, final=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.warning(f"Cancel called for task {context.task_id}, but not supported.")
        await TaskUpdater(event_queue, context.task_id, context.context_id).update_status(
            TaskState.failed, final=True
        )