# File: a2a/server/agent_execution.py
from abc import ABC, abstractmethod
from a2a.types import TaskStatusUpdateEvent, TaskArtifactUpdateEvent

class AgentExecutor(ABC):
    @abstractmethod
    async def execute(self, context, event_queue):
        pass

    async def cancel(self, context, event_queue):
        raise NotImplementedError("Cancel not supported")
