# server.py
import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# import helper khởi tạo HostAgent từ file routing_agent.py của bạn
from host_agent.routing_agent import get_initialized_routing_agent_sync  # dùng file bạn có. :contentReference[oaicite:5]{index=5}

logger = logging.getLogger("webapp")
logging.basicConfig(level=logging.INFO)

# Địa chỉ remote agents - giữ giống config bạn đang dùng ở gradio_app.py
remote_agent_addresses = [
    "http://localhost:10001",  # SymptomAgent
    "http://localhost:10002",  # CostAgent
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: cho phép tất cả; production: chỉnh lại domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount static files (css, js, images)
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")

# place to hold HostAgent
app.state.agent = None

class Query(BaseModel):
    message: str

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up: initializing HostAgent (blocking sync init in executor)...")
    loop = asyncio.get_running_loop()
    agent = await loop.run_in_executor(None, get_initialized_routing_agent_sync, remote_agent_addresses)
    app.state.agent = agent
    logger.info("HostAgent initialized and ready.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # trả template base.html; trong template dùng src sinh động: /static/...
    return templates.TemplateResponse("base.html", {"request": request})

@app.post("/chat")
async def chat(query: Query):
    agent = app.state.agent
    if agent is None:
        return JSONResponse({"answer": "[Error] Agent chưa sẵn sàng, thử lại sau."}, status_code=503)
    try:
        # agent.ainvoke là coroutine (đã có trong HostAgent của bạn). :contentReference[oaicite:6]{index=6}
        result = await agent.ainvoke(query.message, session_id="web-session")
        # result có thể là dict / list / str
        if isinstance(result, dict):
            answer = result.get("content") or result.get("text") or str(result)
        elif isinstance(result, list):
            # list thường chứa string hoặc dict parts; flatten
            parts = []
            for r in result:
                if isinstance(r, dict):
                    parts.append(r.get("text") or str(r))
                else:
                    parts.append(str(r))
            answer = "\n\n".join(parts)
        else:
            answer = str(result)
    except Exception as e:
        logger.exception("Error when calling agent.ainvoke")
        answer = f"[Error] Exception while invoking agent: {e}"
    return {"answer": answer}

@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": app.state.agent is not None}
