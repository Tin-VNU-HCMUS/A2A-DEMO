from typing import Any
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, Part
from a2a.utils.message import new_agent_text_message

from symptom_agent import SymptomAgent, ResponseFormat  # Import đúng agent
from agents.symptom_agent.formatter import format_response_with_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SymptomAgentExecutor(AgentExecutor):
    def __init__(self, mcp_tools: list[Any]):
        self.agent = SymptomAgent(mcp_tools=mcp_tools)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> ResponseFormat:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            await updater.update_status(TaskState.submitted)

        await updater.update_status(TaskState.working)

        query = context.get_user_input()
        last_content: str | None = None

        try:
            async for event in self.agent.stream(query, context.context_id):
                content = event.get("content") if isinstance(event, dict) else str(event)
                last_content = content
                await updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(content),
                )

            # gửi artifact cuối cùng nếu có nội dung
            if last_content:
                # ⚡ Lấy data từ event cuối cùng (hoặc None)
                try:
                    data = event.get("data") if isinstance(event, dict) else None
                except Exception:
                    data = None

                final_text = format_response_with_llm(last_content, data)

                parts = [Part(root=TextPart(text=final_text))]
                logger.debug(
                    "[SymptomAgentExecutor] Final artifact parts dump: %s",
                    parts,
                )
                await updater.add_artifact(parts)

            await updater.update_status(TaskState.completed, final=True)

            # Trả về ResponseFormat với message là kết quả thực tế
            
            #return ResponseFormat(
                #status="completed",
                #message=last_content or "Hoàn thành",
                #data=None
            #)
            
            return ResponseFormat(
                status="completed",
                message=final_text or "Hoàn thành",
                data=data
            )

        except Exception as e:
            logger.error(f"SymptomAgentExecutor error: {e}", exc_info=True)
            await updater.update_status(TaskState.failed, final=True)

            # Luôn trả ResponseFormat thay vì None
            return ResponseFormat(
                status="failed",
                message=f"Lỗi: {str(e)}",
                data=None
            )

    async def cancel(self, context, event_queue):
        raise Exception("Cancel not supported")
