# a2a/utils/task_factory.py

import uuid
from a2a.types import Task
from a2a.utils.message_factory import new_agent_text_message
from a2a.types import SendMessageRequest

def new_task(context, message: SendMessageRequest) -> Task:
    return Task(
        id=str(uuid.uuid4()),
        thread_id=message.threadId or str(uuid.uuid4()),
        sender=context.card,
        message=new_agent_text_message(message, context.card.id),
        context_id=str(uuid.uuid4())
    )



