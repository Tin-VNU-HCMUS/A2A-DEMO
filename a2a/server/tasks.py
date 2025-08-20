# File: a2a/server/tasks.py
class InMemoryTaskStore:
    def __init__(self):
        self.tasks = {}

    async def save(self):
        pass

    async def load(self):
        pass