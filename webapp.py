# webapp.py
from __future__ import annotations

import os
import uuid
import json
import threading
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import Agent

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
UPLOADS_DIR = os.path.join(WORKSPACE_DIR, "uploads")

os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# In-memory task status store
TASKS: Dict[str, Dict[str, Any]] = {}


class TaskReq(BaseModel):
    goal: str
    model: Optional[str] = "gpt-oss:20b"
    base_url: Optional[str] = "http://localhost:11434"
    workspace: Optional[str] = WORKSPACE_DIR


class TaskResp(BaseModel):
    task_id: str
    status: str


def run_task(task_id: str, req: TaskReq):
    TASKS[task_id]["status"] = "running"
    try:
        agent = Agent(workspace=req.workspace or WORKSPACE_DIR, model=req.model or "gpt-oss:20b", base_url=req.base_url or "http://localhost:11434")
        result = agent.run(req.goal)
        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["result"] = result
    except Exception as e:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Missing static/index.html</h1>"


@app.post("/api/task", response_model=TaskResp)
def create_task(req: TaskReq):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "queued", "req": req.model_dump(), "result": None, "error": None}

    t = threading.Thread(target=run_task, args=(task_id, req), daemon=True)
    t.start()
    return TaskResp(task_id=task_id, status="queued")


@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="task not found")
    return TASKS[task_id]


@app.get("/api/workspace_files")
def workspace_files(path: str = ""):
    target = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not target.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise HTTPException(status_code=400, detail="invalid path")
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="path not found")

    if os.path.isdir(target):
        items = []
        for name in sorted(os.listdir(target)):
            fp = os.path.join(target, name)
            items.append({
                "name": name,
                "is_dir": os.path.isdir(fp),
                "size": os.path.getsize(fp),
            })
        return {"path": path, "items": items}

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {"path": path, "is_dir": False, "content": content}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(UPLOADS_DIR, safe_name)
    data = await file.read()
    with open(dest, "wb") as f:
        f.write(data)
    return {"ok": True, "filename": file.filename, "saved_as": os.path.relpath(dest, WORKSPACE_DIR)}
