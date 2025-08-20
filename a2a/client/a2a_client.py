import httpx
from a2a.types import AgentCard, SendMessageRequest, SendMessageResponse

class A2AClient:
    def __init__(self, http_client: httpx.AsyncClient, card: AgentCard, url: str):
        self.card = card
        self.url = url.rstrip("/")
        self.http_client = http_client

    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        response = await self.http_client.post(f"{self.url}/messages", json=request.model_dump())
        response.raise_for_status()
        return SendMessageResponse(**response.json())