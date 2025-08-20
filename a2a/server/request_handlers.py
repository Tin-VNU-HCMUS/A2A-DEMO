# File: a2a/server/request_handlers.py
from starlette.requests import Request
from starlette.responses import JSONResponse

class DefaultRequestHandler:
    def __init__(self, agent_executor, task_store):
        self.agent_executor = agent_executor
        self.task_store = task_store

    @property
    def routes(self):
        from starlette.routing import Route
        return [
            Route("/messages", self.handle_message, methods=["POST"]),
        ]

    async def handle_message(self, request: Request):
        body = await request.json()
        # Fake simple logic for prototype
        return JSONResponse({"status": "received", "echo": body})