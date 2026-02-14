# tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional
import os
import re
import ipaddress
import socket
import subprocess
import urllib.request
import urllib.parse
from urllib.error import URLError, HTTPError

from memory_store import MemoryStore


class ToolError(Exception):
    pass


# -----------------------------
# Path safety (workspace sandbox)
# -----------------------------
def _normalize_rel_path(path: str) -> str:
    """
    Normalize user/model-provided path into a safe workspace-relative path.

    Behaviors:
    - Strip leading slashes so "/a/b.txt" becomes "a/b.txt"
    - Normalize "." and ".."
    - Reject traversal that escapes workspace
    """
    if path is None:
        return ""

    p = str(path).strip()
    p = p.lstrip("/\\")
    p = p.replace("\\", "/")
    p = os.path.normpath(p)

    if p == ".":
        p = ""

    # Reject traversal remnants
    if p.startswith("..") or "/.." in p.replace("\\", "/"):
        raise ToolError(f"Invalid path traversal: {path}")

    return p


def _resolve_in_workspace(workspace: str, path: str) -> str:
    """
    Resolve path inside workspace; prevent path traversal.
    Accept absolute-looking paths ("/foo.txt") by normalizing to workspace-relative.
    """
    workspace_abs = os.path.abspath(workspace)
    rel = _normalize_rel_path(path)
    p = os.path.abspath(os.path.join(workspace_abs, rel))

    if not (p == workspace_abs or p.startswith(workspace_abs + os.sep)):
        raise ToolError(f"Path escapes workspace: {path}")
    return p


# -----------------------------
# HTTP fetch (already in your file)
# -----------------------------
def _is_private_host(host: str) -> bool:
    """Resolve host and decide whether it points to private/loopback/link-local addresses."""
    if not host:
        return False

    # Raw IP literal?
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        # If cannot resolve, treat as non-private; fetch will fail later anyway.
        return False

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
        except Exception:
            continue
    return False


def http_fetch(
    workspace: str,
    url: str,
    method: str = "GET",
    headers: Dict[str, str] | None = None,
    data: str | None = None,
    timeout: int = 15,
    max_bytes: int = 200_000,
    allow_private: bool = False,
) -> Dict[str, Any]:
    """Fetch a URL over HTTP/HTTPS and return text content."""
    if not url or not isinstance(url, str):
        raise ToolError("url must be a non-empty string")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError("Only http/https URLs are allowed")

    host = parsed.hostname or ""
    if not allow_private and _is_private_host(host):
        raise ToolError(
            "Refusing to fetch private/loopback/link-local address (set allow_private=true to override)"
        )

    req_headers = {
        "User-Agent": "LocalAgent/1.0",
        "Accept": "*/*",
    }
    if headers:
        for k, v in headers.items():
            if any(ch in str(k) for ch in ["\r", "\n"]) or any(ch in str(v) for ch in ["\r", "\n"]):
                raise ToolError("Invalid header: contains newline")
        req_headers.update({str(k): str(v) for k, v in headers.items()})

    body: bytes | None = None
    if data is not None:
        body = data.encode("utf-8")

    m = (method or "GET").upper().strip()
    if m not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
        raise ToolError(f"Unsupported method: {method}")

    req = urllib.request.Request(url=url, data=body, headers=req_headers, method=m)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            final_url = resp.geturl()
            resp_headers = dict(resp.headers.items())

            raw = resp.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            if truncated:
                raw = raw[:max_bytes]

            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = raw.decode("latin-1", errors="replace")

            return {
                "url": url,
                "final_url": final_url,
                "status": int(status),
                "headers": {k: resp_headers.get(k) for k in list(resp_headers.keys())[:50]},
                "content": text,
                "truncated": truncated,
                "bytes": len(raw),
            }
    except HTTPError as e:
        try:
            err_body = e.read(max_bytes + 1)
        except Exception:
            err_body = b""
        truncated = len(err_body) > max_bytes
        if truncated:
            err_body = err_body[:max_bytes]
        try:
            text = err_body.decode("utf-8", errors="replace")
        except Exception:
            text = err_body.decode("latin-1", errors="replace")
        return {
            "url": url,
            "final_url": getattr(e, "url", url),
            "status": int(getattr(e, "code", 0) or 0),
            "headers": dict(getattr(e, "headers", {}) or {}),
            "content": text,
            "truncated": truncated,
            "bytes": len(err_body),
            "error": str(e),
        }
    except URLError as e:
        raise ToolError(f"Network error: {e}")
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}")


# -----------------------------
# File tools
# -----------------------------
def read_file(workspace: str, path: str, max_bytes: int = 200_000) -> Dict[str, Any]:
    p = _resolve_in_workspace(workspace, path)
    if not os.path.exists(p):
        raise ToolError(f"File not found: {path}")
    with open(p, "rb") as f:
        data = f.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    return {"path": _normalize_rel_path(path), "content": text, "truncated": truncated}


def write_file(
    workspace: str,
    path: str,
    content: str,
    overwrite: bool = True,
    create_dirs: bool = True,
) -> Dict[str, Any]:
    p = _resolve_in_workspace(workspace, path)
    if os.path.exists(p) and not overwrite:
        raise ToolError(f"Refusing to overwrite existing file: {path}")

    parent = os.path.dirname(p)
    if create_dirs:
        os.makedirs(parent, exist_ok=True)
    else:
        if parent and not os.path.exists(parent):
            raise ToolError(f"Parent directory does not exist: {parent}")

    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    return {"path": _normalize_rel_path(path), "bytes": len(content.encode("utf-8"))}


def append_file(workspace: str, path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
    p = _resolve_in_workspace(workspace, path)
    parent = os.path.dirname(p)
    if create_dirs:
        os.makedirs(parent, exist_ok=True)
    else:
        if parent and not os.path.exists(parent):
            raise ToolError(f"Parent directory does not exist: {parent}")

    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(content)

    return {"path": _normalize_rel_path(path), "bytes_appended": len(content.encode("utf-8"))}


def list_dir(workspace: str, path: str = "") -> Dict[str, Any]:
    p = _resolve_in_workspace(workspace, path)
    if not os.path.exists(p):
        raise ToolError(f"Directory not found: {path}")
    if not os.path.isdir(p):
        raise ToolError(f"Not a directory: {path}")

    items = []
    for name in sorted(os.listdir(p)):
        fp = os.path.join(p, name)
        st = os.stat(fp)
        items.append(
            {"name": name, "is_dir": os.path.isdir(fp), "size": st.st_size, "mtime": st.st_mtime}
        )
    return {"path": _normalize_rel_path(path), "items": items}


def apply_patch(workspace: str, path: str, patch: str, create_if_missing: bool = False) -> Dict[str, Any]:
    """
    Minimal patch helper with exact match replacement blocks:

    ---BEGIN_OLD---
    <old text>
    ---END_OLD---
    ---BEGIN_NEW---
    <new text>
    ---END_NEW---

    If OLD block not found exactly, it fails.
    """
    p = _resolve_in_workspace(workspace, path)

    if not os.path.exists(p):
        if create_if_missing:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write("")
        else:
            raise ToolError(f"File not found: {path}")

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    patch = patch.replace("\r\n", "\n").replace("\r", "\n")

    def _extract(tag: str) -> str:
        m = re.search(rf"---BEGIN_{tag}---\n(.*?)\n---END_{tag}---", patch, re.DOTALL)
        if not m:
            raise ToolError(f"Patch missing {tag} block")
        return m.group(1)

    old = _extract("OLD")
    new = _extract("NEW")

    if old not in text:
        raise ToolError("OLD block not found in file (exact match required)")

    text2 = text.replace(old, new, 1)

    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text2)

    return {"path": _normalize_rel_path(path), "bytes": os.path.getsize(p)}


# -----------------------------
# NEW: Bash tool (runs in workspace)
# -----------------------------
_DANGEROUS_PATTERNS = [
    r"\bsudo\b",
    r"\brm\s+-rf\s+/\b",
    r"\brm\s+-rf\s+--no-preserve-root\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
]


def run_bash(
    workspace: str,
    cmd: str,
    timeout_sec: int = 60,
    max_output: int = 200_000,
) -> Dict[str, Any]:
    """
    Run a bash command with cwd pinned to workspace.
    Minimal safety checks to prevent obvious foot-guns.

    Args:
      cmd: shell command string
      timeout_sec: execution timeout
      max_output: truncate stdout/stderr
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ToolError("cmd must be a non-empty string")

    c = cmd.strip()

    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, c):
            raise ToolError(f"Refusing potentially dangerous command (matched: {pat})")

    ws = os.path.abspath(workspace)

    try:
        proc = subprocess.run(
            c,
            shell=True,
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout_sec}s")
    except Exception as e:
        raise ToolError(f"Command failed to start: {type(e).__name__}: {e}")

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    out_trunc = False
    err_trunc = False

    if len(stdout) > max_output:
        stdout = stdout[:max_output]
        out_trunc = True
    if len(stderr) > max_output:
        stderr = stderr[:max_output]
        err_trunc = True

    return {
        "cmd": c,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": out_trunc,
        "stderr_truncated": err_trunc,
    }


# -----------------------------
# Memory tools
# -----------------------------
def _mem(workspace: str) -> MemoryStore:
    db_path = os.path.join(os.path.abspath(workspace), "memory.sqlite")
    return MemoryStore(db_path)


def mem_put(
    workspace: str,
    key: str,
    type: str,
    content: Any,
    tags: list[str] | None = None,
    source: str = "model",
    confidence: float = 0.6,
    reason: str = "",
    overwrite: bool = True,
    note: str = "",
):
    store = _mem(workspace)
    return store.put(
        actor="agent:model",
        key=key,
        type=type,
        content=content,
        tags=tags,
        source=source,
        confidence=confidence,
        reason=reason,
        overwrite=overwrite,
        note=note,
    )


def mem_get(workspace: str, key: str, include_deleted: bool = False):
    store = _mem(workspace)
    return store.get(key=key, include_deleted=include_deleted)


def mem_search(
    workspace: str,
    q: str = "",
    type: str | None = None,
    tag: str | None = None,
    limit: int = 20,
    include_deleted: bool = False,
):
    store = _mem(workspace)
    return store.search(q=q, type=type, tag=tag, limit=limit, include_deleted=include_deleted)


def mem_update(
    workspace: str,
    item_id: int,
    content: Any,
    tags: list[str] | None = None,
    type: str | None = None,
    source: str = "model",
    confidence: float = 0.6,
    reason: str = "",
    note: str = "",
):
    store = _mem(workspace)
    return store.update(
        actor="agent:model",
        item_id=item_id,
        content=content,
        tags=tags,
        type=type,
        source=source,
        confidence=confidence,
        reason=reason,
        note=note,
    )


def mem_delete(workspace: str, item_id: int, reason: str = ""):
    store = _mem(workspace)
    return store.delete(actor="agent:model", item_id=item_id, reason=reason)


def mem_history(workspace: str, item_id: int, limit: int = 20):
    store = _mem(workspace)
    return store.history(item_id=item_id, limit=limit)


def mem_revert(workspace: str, item_id: int, rev_id: int, reason: str = ""):
    store = _mem(workspace)
    return store.revert(actor="agent:model", item_id=item_id, rev_id=rev_id, reason=reason)


def mem_dump(workspace: str, limit: int = 200, include_deleted: bool = True):
    store = _mem(workspace)
    return store.search(q="", type=None, tag=None, limit=limit, include_deleted=include_deleted)


# -----------------------------
# Tool registry
# -----------------------------
@dataclass
class ToolSpec:
    name: str
    handler: Callable[..., Any]
    description: str


def get_tool_registry() -> Dict[str, ToolSpec]:
    return {
        "read_file": ToolSpec(
            name="read_file",
            handler=read_file,
            description="Read a UTF-8 text file from workspace. Args: path, max_bytes(optional).",
        ),
        "write_file": ToolSpec(
            name="write_file",
            handler=write_file,
            description="Write a UTF-8 text file to workspace. Args: path, content, overwrite(optional), create_dirs(optional).",
        ),
        "list_dir": ToolSpec(
            name="list_dir",
            handler=list_dir,
            description="List directory under workspace. Args: path(optional).",
        ),
        "append_file": ToolSpec(
            name="append_file",
            handler=append_file,
            description="Append UTF-8 text to a file in workspace. Args: path, content, create_dirs(optional).",
        ),
        "apply_patch": ToolSpec(
            name="apply_patch",
            handler=apply_patch,
            description="Apply a simple exact-match patch. Args: path, patch, create_if_missing(optional).",
        ),
        "run_bash": ToolSpec(
            name="run_bash",
            handler=run_bash,
            description="Run a bash command in workspace. Args: cmd, timeout_sec(optional), max_output(optional).",
        ),
        "http_fetch": ToolSpec(
            name="http_fetch",
            handler=http_fetch,
            description="Fetch a URL over http/https. Args: url, method?, headers?, data?, timeout?, max_bytes?, allow_private?",
        ),
        "mem_put": ToolSpec(
            name="mem_put",
            handler=mem_put,
            description="Upsert a memory by key. Args: key,type,content,tags?,source?,confidence?,reason?,overwrite?,note?",
        ),
        "mem_get": ToolSpec(
            name="mem_get",
            handler=mem_get,
            description="Get current memory by key. Args: key, include_deleted?",
        ),
        "mem_search": ToolSpec(
            name="mem_search",
            handler=mem_search,
            description="Search memories. Args: q?, type?, tag?, limit?, include_deleted?",
        ),
        "mem_update": ToolSpec(
            name="mem_update",
            handler=mem_update,
            description="Update memory by item_id. Args: item_id, content, tags?, type?, source?, confidence?, reason?, note?",
        ),
        "mem_delete": ToolSpec(
            name="mem_delete",
            handler=mem_delete,
            description="Soft delete memory by item_id. Args: item_id, reason?",
        ),
        "mem_history": ToolSpec(
            name="mem_history",
            handler=mem_history,
            description="List version history. Args: item_id, limit?",
        ),
        "mem_revert": ToolSpec(
            name="mem_revert",
            handler=mem_revert,
            description="Revert to historical version. Args: item_id, rev_id, reason?",
        ),
        "mem_dump": ToolSpec(
            name="mem_dump",
            handler=mem_dump,
            description="Dump recent memories. Args: limit?, include_deleted?",
        ),
    }
