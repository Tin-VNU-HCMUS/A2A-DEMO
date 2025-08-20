# agents/booking_agent/agent_executor.py
from typing import Any
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TaskArtifactUpdateEvent, TaskStatusUpdateEvent, TaskState, TaskStatus
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from agents.booking_agent.booking_agent import BookingAgent

class BookingAgentExecutor(AgentExecutor):
    def __init__(self, mcp_tools: list[Any]):
        self.agent = BookingAgent(mcp_tools)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context, query)
        await event_queue.enqueue_event(task)
        async for event in self.agent.stream(query, task.contextId):
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(state=TaskState.input_required),
                    message=new_agent_text_message(event["content"], task.agentMessageId)
                )
            )
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                artifact=new_text_artifact(event["content"]),
                is_final=True
            )
        )

    async def cancel(self, context, event_queue):
        raise Exception("Cancel not supported")
