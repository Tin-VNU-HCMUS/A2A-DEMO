from typing import Any
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, Part, DataPart
from a2a.utils.message import new_agent_text_message

from symptom_agent import SymptomAgent, ResponseFormat  # Import đúng agent


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
        data: Any | None = None

        try:
            async for event in self.agent.stream(query, context.context_id):
                # Robust extraction of content and possible data payload
                if isinstance(event, dict):
                    content = event.get("content") if event.get("content") is not None else str(event)
                    # prefer explicit 'data' field if present
                    if "data" in event:
                        data = event.get("data")
                else:
                    # fallback for non-dict event objects
                    content = getattr(event, "content", None) or getattr(event, "text", None) or str(event)

                last_content = content

                # Update streaming status so host can forward partial text to clients
                await updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(content),
                )

            # gửi artifact cuối cùng nếu có nội dung (hoặc gửi artifact rỗng kèm data nếu chỉ có data)
            final_text = last_content or ""

            # Lấy data đã thu được (nếu có)
            # data đã được gán trong loop nếu event dict chứa 'data'

            artifact_parts: list[Part] = []
            # luôn thêm text part (dễ hiển thị cho client)
            artifact_parts.append(Part(root=TextPart(text=final_text)))

            # nếu có structured data, đính kèm như DataPart để HostAgent có thể forward nguyên structure
            if data is not None:
                artifact_parts.append(Part(root=DataPart(data=data)))

            logger.debug(
                "[SymptomAgentExecutor] Final artifact parts dump: %s",
                artifact_parts,
            )

            if artifact_parts:
                await updater.add_artifact(artifact_parts)

            await updater.update_status(TaskState.completed, final=True)

            # Trả về ResponseFormat với message là kết quả thực tế
            return ResponseFormat(
                status="completed",
                message=final_text or "Hoàn thành",
                data=data,
            )

        except Exception as e:
            logger.error(f"SymptomAgentExecutor error: {e}", exc_info=True)
            await updater.update_status(TaskState.failed, final=True)

            # Luôn trả ResponseFormat thay vì None
            return ResponseFormat(
                status="error",
                message=f"Lỗi: {str(e)}",
                data=None,
            )

    async def cancel(self, context, event_queue):
        raise Exception("Cancel not supported")
