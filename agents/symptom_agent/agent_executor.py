from typing import Literal, Any, AsyncIterable
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskState,
    TaskStatus,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
    new_text_artifact,
)
from symptom_agent import SymptomAgent  # Nhớ import đúng agent



class SymptomAgentExecutor(AgentExecutor):
    def __init__(self, mcp_tools: list[Any]):
        self.agent = SymptomAgent(mcp_tools=mcp_tools)

    # Hàm xử lý task được gửi từ A2A Server
    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        # Nếu chưa có task, tạo mới
        if not task:
            task = new_task(context, query)
            await event_queue.enqueue_event(task)

        # Stream phản hồi từng phần từ agent
        async for event in self.agent.stream(query, task.contextId):
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(state=TaskState.input_required),
                    message=new_agent_text_message(event["content"], task.agentMessageId)
                )
            )

        # Gửi phản hồi cuối cùng
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                artifact=new_text_artifact(event["content"]),
                is_final=True
            )
        )

    # Không hỗ trợ hủy task
    async def cancel(self, context, event_queue):
        raise Exception("Cancel not supported")
