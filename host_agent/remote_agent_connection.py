from collections.abc import Callable
from uuid import uuid4
from a2a.types import TaskState

import httpx

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

import logging
logger = logging.getLogger("RemoteAgentConnections")
logger.setLevel(logging.DEBUG)



TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, client: httpx.AsyncClient, agent_card: AgentCard):
        self.agent_client = A2AClient(client, agent_card)
        self.card = agent_card
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self,
        request: MessageSendParams,
        task_callback: TaskUpdateCallback | None,
    ) -> Task | Message | None:
        """
        Send message to remote agent. For streaming-capable agents we accumulate
        streaming events into a lightweight Task-like object and return it at the end.
        We still call provided task_callback(event, card) for each event so callers
        receiving callbacks can update live state. This prevents returning None.
        """
        # helper: make lightweight task-like object
        def _make_task_like(task_id: str):
            from types import SimpleNamespace
            t = SimpleNamespace()
            t.id = task_id
            t.contextId = None
            t.artifacts = []
            # status will be a SimpleNamespace with at least .state and optional .message
            t.status = SimpleNamespace()
            t.status.state = None
            t.status.message = None
            return t

        # Use streaming path if supported
        if self.card.capabilities.streaming:
            accumulated_task = None
            last_task_id = None
            try:
                async for response in self.agent_client.send_message_streaming(
                    SendStreamingMessageRequest(id=str(uuid4()), params=request)
                ):
                    # Defensive access to root/result/error
                    root = getattr(response, "root", None)
                    if root is None:
                        logger.debug("[RemoteAgentConnections] streaming yielded no root, skipping")
                        continue

                    # If RPC-level error
                    rpc_error = getattr(root, "error", None)
                    result = getattr(root, "result", None)
                    if not result and rpc_error:
                        logger.error("[RemoteAgentConnections] RPC error in streaming response: %s", rpc_error)
                        # return the error structure as caller previously did
                        return rpc_error

                    # If remote returned a Message -> immediate end
                    if isinstance(result, Message):
                        logger.debug("[RemoteAgentConnections] streaming returned a Message (final).")
                        return result

                    # result is an event (TaskStatusUpdateEvent / TaskArtifactUpdateEvent or dict)
                    event = result

                    # Determine task id from event if available
                    task_id = None
                    if hasattr(event, "taskId"):
                        task_id = getattr(event, "taskId")
                    elif isinstance(event, dict):
                        task_id = event.get("taskId") or event.get("task_id")

                    if not task_id:
                        # fallback to request.taskId or generate
                        task_id = getattr(request.message, "taskId", None) or str(uuid4())

                    last_task_id = task_id
                    if accumulated_task is None:
                        accumulated_task = _make_task_like(task_id)

                    # update contextId if present
                    context_id = getattr(event, "contextId", None) or (event.get("contextId") if isinstance(event, dict) else None)
                    if context_id:
                        accumulated_task.contextId = context_id

                    # Update status if present
                    status = getattr(event, "status", None) or (event.get("status") if isinstance(event, dict) else None)
                    if status:
                        # status might be object-like or dict
                        state_val = getattr(status, "state", None) or (status.get("state") if isinstance(status, dict) else None)
                        if state_val is not None:
                            accumulated_task.status.state = state_val
                        msg = getattr(status, "message", None) or (status.get("message") if isinstance(status, dict) else None)
                        if msg:
                            accumulated_task.status.message = msg

                    # Artifact handling
                    artifact = getattr(event, "artifact", None) or (event.get("artifact") if isinstance(event, dict) else None)
                    if artifact:
                        accumulated_task.artifacts.append(artifact)

                    # invoke caller callback (if any) so HostAgent can update realtime state
                    if task_callback:
                        try:
                            # callback may return a Task-like object; we ignore/keep accumulated_task
                            cb_ret = task_callback(event, self.card)
                            if cb_ret is not None:
                                # if callback returns a richer Task object, prefer it as final accumulator
                                accumulated_task = cb_ret
                        except Exception as cb_exc:
                            logger.exception("[RemoteAgentConnections] task_callback raised: %s", cb_exc)

                    # If event signals final, break
                    final_flag = getattr(event, "final", None)
                    if final_flag is None and isinstance(event, dict):
                        final_flag = event.get("final")
                    if final_flag:
                        logger.debug("[RemoteAgentConnections] Received final==True, finishing streaming.")
                        break

                # end async for
            except Exception as e:
                logger.exception("[RemoteAgentConnections] exception during streaming: %s", e)
                # If we have an accumulated task, mark failed; else propagate the exception
                if accumulated_task:
                    accumulated_task.status.state = getattr(TaskState, "failed", "failed")
                    return accumulated_task
                raise

            # If we never accumulated anything, return empty task-like so caller doesn't get None
            if accumulated_task is None:
                accumulated_task = _make_task_like(last_task_id or str(uuid4()))
                accumulated_task.status.state = getattr(TaskState, "completed", "completed")
            return accumulated_task

        # Non-streaming (existing logic)
        response = await self.agent_client.send_message(
            SendMessageRequest(id=str(uuid4()), params=request)
        )
        if isinstance(response.root, JSONRPCErrorResponse):
            return response.root.error
        if isinstance(response.root.result, Message):
            return response.root.result

        if task_callback:
            try:
                task_callback(response.root.result, self.card)
            except Exception:
                logger.exception("[RemoteAgentConnections] task_callback raised for non-streaming response")
        return response.root.result
