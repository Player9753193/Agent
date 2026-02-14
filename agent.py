# agent.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import re
import os
import requests

from ollama_client import OllamaClient
from tools import get_tool_registry, ToolError


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """\
You are an autonomous local coding agent running on a user's machine.

MEMORY POLICY:
- Before planning, ALWAYS search memory for relevant info using mem_search(q=...).
  - Use memory results to reuse prior decisions/conventions and avoid redoing work.

- What to store (use mem_put / mem_update):
  - Stable facts (fact.*), user preferences (pref.*), project decisions/conventions (decision.*),
    and produced artifacts/paths (artifact.*).
  - Store only high-signal, reusable summaries; avoid raw logs and long transcripts.

- Required fields & naming:
  - Each stored memory content MUST include:
    - recorded_at: ISO8601 timestamp
    - why: short reason why this is worth remembering
    - value: the remembered information (structured JSON preferred)
  - Prefer stable keys: decision.<topic>, pref.<topic>, fact.<topic>, artifact.<topic>

- Safety / privacy:
  - Do NOT store secrets (API keys, passwords, tokens) or sensitive personal data in memory.

TOOLS (ONLY):
- list_dir(path?)
- read_file(path, max_bytes?)
- write_file(path, content, overwrite?, create_dirs?)
- append_file(path, content, create_dirs?)
- apply_patch(path, patch, create_if_missing?)
- run_bash(cmd, timeout_sec?, max_output?)       (runs in workspace)

CRITICAL RULES:
- You MUST respond with exactly ONE JSON object and no extra text.
- Allowed response types:
  1) PLAN:
     {"type":"plan","goal":"...","tasks":[{"id":"T1","title":"...","tool_hints":["list_dir"],"done":false}], "acceptance":["..."]}
  2) STEP:
     {"type":"step","task_id":"T1","action":{"name":"read_file","args":{"path":"README.md"}}, "check":"..."}
  3) FINAL:
     {"type":"final","summary":"...","artifacts":[{"path":"...","desc":"..."}], "notes":["..."]}

- Always create a plan first unless the goal is trivial.
- Execute tasks one at a time. After each tool result, decide next step.
- Prefer reading existing files before overwriting. Use apply_patch when editing existing content.
- Never access paths outside workspace.
"""


def _extract_last_json_object(text: str) -> Dict[str, Any]:
    matches = JSON_OBJ_RE.findall(text)
    if not matches:
        raise ValueError("No JSON object found.")
    return json.loads(matches[-1])


class Agent:
    def __init__(
        self,
        model: str = "gpt-oss:20b",
        workspace: str = "./workspace",
        base_url: str = "http://localhost:11434",
        max_steps: int = 128,
        verbose: bool = True,
        summary_model: str = "qwen3:0.6b",
        summary_trigger_chars: int = 35000,
        summary_keep_last: int = 10,
    ):
        self.model = model
        self.workspace = workspace
        self.max_steps = max_steps
        self.verbose = verbose

        self.summary_model = summary_model
        self.summary_trigger_chars = int(summary_trigger_chars)
        self.summary_keep_last = int(summary_keep_last)
        self._context_summary = ""

        self.client = OllamaClient(base_url=base_url)
        self.tools = get_tool_registry()

    def _messages_char_count(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for m in messages:
            total += len((m.get("role") or "")) + len((m.get("content") or ""))
        return total

    def _summarize_with_small_model(self, text: str) -> str:
        prompt = (
            "你是一个压缩上下文的摘要器。把输入压缩成可复用的“工作记忆”，要求：\n"
            "- 只保留：用户目标、已完成事项、关键约束/假设、重要文件路径、下一步可执行计划。\n"
            "- 不要保留冗长日志、逐字输出、无关细节。\n"
            "- 用中文，条目式，长度尽量短（<= 500 字）。\n\n"
            "INPUT:\n"
            f"{text}"
        )

        resp = self.client.chat(
            model=self.summary_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"num_predict": 512, "num_ctx": 2048},
        )
        content = ((resp.get("message") or {}).get("content")) or ""
        return content.strip()

    def _maybe_compact_context(self, messages: List[Dict[str, str]], task_text: str) -> List[Dict[str, str]]:
        if self._messages_char_count(messages) < self.summary_trigger_chars:
            return messages

        keep_n = max(4, self.summary_keep_last)
        history = messages[:-keep_n]
        tail = messages[-keep_n:]

        history_text = "\n".join(f"[{m.get('role','')}] {m.get('content','')}" for m in history)

        summary = self._summarize_with_small_model(history_text)
        self._context_summary = summary

        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        rebuilt: List[Dict[str, str]] = []
        if system_msg:
            rebuilt.append(system_msg)

        rebuilt.append({"role": "user", "content": task_text})
        rebuilt.append({"role": "assistant", "content": "CONTEXT_SUMMARY:\n" + (summary or "(empty summary)")})
        rebuilt.extend(tail)
        return rebuilt

    def _call_model(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            resp = self.client.chat(model=self.model, messages=messages, stream=False)
        except requests.HTTPError as e:
            if "500" in str(e):
                compacted = self._maybe_compact_context(
                    messages,
                    task_text=messages[1]["content"] if len(messages) > 1 else "",
                )
                resp = self.client.chat(model=self.model, messages=compacted, stream=False)
                messages = compacted
            else:
                raise

        content = resp.get("message", {}).get("content", "")

        if self.verbose:
            print("\n--- MODEL RAW ---")
            print(content if content is not None else "<None>")

        try:
            obj = _extract_last_json_object(content or "")
        except Exception as e:
            return {"type": "__invalid__", "error": str(e), "raw": content}

        if self.verbose:
            print("\n--- MODEL JSON ---")
            print(json.dumps(obj, ensure_ascii=False, indent=2))
        return obj

    def _run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {"ok": False, "tool": name, "error": f"Unknown tool. Available: {list(self.tools.keys())}"}

        # IMPORTANT defaults to avoid "can't overwrite"
        if name == "write_file":
            if "overwrite" not in args:
                args["overwrite"] = True
            if "create_dirs" not in args:
                args["create_dirs"] = True
        if name == "append_file":
            if "create_dirs" not in args:
                args["create_dirs"] = True

        try:
            tool = self.tools[name]
            result = tool.handler(self.workspace, **args)
            return {"ok": True, "tool": name, "result": result}
        except ToolError as te:
            return {"ok": False, "tool": name, "error": str(te)}
        except TypeError as te:
            return {"ok": False, "tool": name, "error": f"Bad args: {te}"}
        except Exception as e:
            return {"ok": False, "tool": name, "error": f"Unexpected error: {e}"}

    def run(self, goal: str) -> Dict[str, Any]:
        os.makedirs(self.workspace, exist_ok=True)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]

        plan: Optional[Dict[str, Any]] = None

        for _step in range(1, self.max_steps + 1):
            messages = self._maybe_compact_context(messages, task_text=goal)

            obj = self._call_model(messages)
            t = obj.get("type")

            if t == "__invalid__":
                messages.append({"role": "assistant", "content": obj.get("raw", "")})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was NOT valid JSON. "
                            "You MUST reply with exactly ONE JSON object and no extra text. "
                            "Return a STEP to execute the first task now."
                        ),
                    }
                )
                continue

            if t == "plan":
                plan = obj
                messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
                messages.append({"role": "user", "content": "Plan received. Start executing the first task with a STEP."})
                continue

            if t == "final":
                return obj

            if t != "step":
                messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
                messages.append(
                    {
                        "role": "user",
                        "content": "Invalid type. Must be one of: plan, step, final. Output JSON only.",
                    }
                )
                continue

            action = obj.get("action") or {}
            tool_name = action.get("name")
            tool_args = action.get("args") or {}

            if not tool_name:
                messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
                messages.append(
                    {"role": "user", "content": "STEP must include action.name. Output JSON only."}
                )
                continue

            tool_out = self._run_tool(tool_name, tool_args)

            messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
            messages.append({"role": "user", "content": "TOOL_RESULT:\n" + json.dumps(tool_out, ensure_ascii=False)})

        return {
            "type": "final",
            "summary": "Max steps reached without finalization.",
            "artifacts": [],
            "notes": ["max_steps reached"],
        }
