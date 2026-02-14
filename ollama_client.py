# ollama_client.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import time
import requests


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


class OllamaClient:
    """
    Minimal Ollama /api/chat client with:
    - configurable connect/read timeouts
    - requests.Session for keep-alive
    - light retry on timeout/connection errors

    Env overrides:
      OLLAMA_CONNECT_TIMEOUT (seconds)
      OLLAMA_READ_TIMEOUT    (seconds)
      OLLAMA_MAX_RETRIES     (int)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        connect_timeout: float = 10.0,
        read_timeout: float = 3600.0,   # <-- 关键：默认从 600 提升到 3600
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

        # allow env override
        self.connect_timeout = _env_float("OLLAMA_CONNECT_TIMEOUT", connect_timeout)
        self.read_timeout = _env_float("OLLAMA_READ_TIMEOUT", read_timeout)
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", str(max_retries)))

        self._sess = requests.Session()

    def _timeout_tuple(self) -> Tuple[float, float]:
        return (self.connect_timeout, self.read_timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout: Optional[Tuple[float, float]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a non-streaming chat request.

        Args:
          messages: list of {"role":"system|user|assistant", "content": "..."}
          temperature: sampling temperature
          timeout: optional (connect_timeout, read_timeout) override
          options: optional dict merged into payload["options"] (e.g., {"num_predict": 1024})
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if options:
            payload["options"].update(options)

        to = timeout if timeout is not None else self._timeout_tuple()

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._sess.post(url, json=payload, timeout=to)
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                # backoff: 0.5s, 1.0s, 2.0s ...
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                # other errors: don't retry
                raise

        # unreachable, but keeps type-checkers happy
        raise RuntimeError(f"Ollama request failed: {last_err}")
