# File: a2a/types/task_status.py
from enum import Enum
from pydantic import BaseModel

class TaskState(str, Enum):
    input_required = "input_required"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"

class TaskStatus(BaseModel):
    state: TaskState

class TaskStatusUpdateEvent(BaseModel):
    status: TaskStatus
    message: str

class TaskArtifactUpdateEvent(BaseModel):
    artifact: dict
    message: str