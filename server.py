# server.py
from __future__ import annotations

import os
import uuid
import json
from typing import Dict, Any, Optional, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import Agent

app = FastAPI()

# 静态目录（index.html 放到 ./static/index.html）
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def extract_json_last(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort: extract the last valid JSON object from a mixed response."""
    t = (text or "").strip()
    if not t:
        return None
    dec = json.JSONDecoder()
    starts = [i for i, ch in enumerate(t) if ch == "{"][-200:]
    for i in reversed(starts):
        frag = t[i:]
        try:
            obj, _end = dec.raw_decode(frag)
            return obj
        except Exception:
            continue
    return None


class ChatReq(BaseModel):
    session_id: str
    user_message: str


class ChatResp(BaseModel):
    assistant_message: str
    memory_written: List[Dict[str, Any]]


# 你可以先用内存存会话；生产建议换 Redis / DB
SESSIONS: Dict[str, List[Dict[str, str]]] = {}


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Missing static/index.html</h1>"


@app.post("/chat")
def chat(req: ChatReq):
    session_id = req.session_id or str(uuid.uuid4())
    msgs = SESSIONS.get(session_id, [])
    msgs.append({"role": "user", "content": req.user_message})

    # NOTE: 这里直接把用户 message 当 goal 跑一遍（简单 demo）
    # 你也可以把 msgs 拼成更完整的 goal/context
    agent = Agent(workspace=os.path.join(os.getcwd(), "workspace"))
    result = agent.run(req.user_message)

    assistant_message = json.dumps(result, ensure_ascii=False, indent=2)
    msgs.append({"role": "assistant", "content": assistant_message})
    SESSIONS[session_id] = msgs

    return JSONResponse({
        "session_id": session_id,
        "assistant_message": assistant_message,
        "memory_written": result.get("notes", []),
    })
