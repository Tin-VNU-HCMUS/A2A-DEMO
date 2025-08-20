# File: a2a/server/apps.py
from starlette.applications import Starlette
from starlette.routing import Route
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

class A2AStarletteApplication:
    def __init__(self, agent_card, request_handler, task_store):
        self.agent_card = agent_card
        self.request_handler = request_handler
        self.task_store = task_store

    def build(self):
        return Starlette(
            routes=self.request_handler.routes,
            on_startup=[self.task_store.load],
            on_shutdown=[self.task_store.save],
        )
