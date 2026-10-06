import os
import json
import threading
from datetime import datetime, timezone

from langchain_core.callbacks.base import BaseCallbackHandler

from bdikit.download import BDIKIT_CACHE_DIR

TRACES_DIR = os.path.join(BDIKIT_CACHE_DIR, "chatbot_traces")


def _safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return _safe(to_dict())
        except Exception:
            pass
    return str(value)


class JSONLTraceLogger(BaseCallbackHandler):
    # One line per event so a task's full trace (reasoning + tool calls) can be replayed later.

    def __init__(self, trace_path, thread_id=None):
        self.trace_path = trace_path
        self.thread_id = thread_id
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    def _write(self, event, run_id, parent_run_id, **fields):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thread_id": self.thread_id,
            "event": event,
            "run_id": str(run_id) if run_id else None,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            **fields,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with open(self.trace_path, "a") as f:
                f.write(line + "\n")

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ):
        self._write(
            "llm_start",
            run_id,
            parent_run_id,
            model=_safe((serialized or {}).get("id")),
            messages=[_safe(m) for batch in messages for m in batch],
        )

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ):
        self._write("llm_start", run_id, parent_run_id, prompts=_safe(prompts))

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._write("llm_end", run_id, parent_run_id, response=_safe(response))

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._write("llm_error", run_id, parent_run_id, error=str(error))

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs
    ):
        self._write(
            "tool_start",
            run_id,
            parent_run_id,
            tool=_safe((serialized or {}).get("name")),
            input=_safe(input_str),
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._write("tool_end", run_id, parent_run_id, output=_safe(output))

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._write("tool_error", run_id, parent_run_id, error=str(error))

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        name = (serialized or {}).get("name") or kwargs.get("name")
        self._write(
            "chain_start", run_id, parent_run_id, name=name, inputs=_safe(inputs)
        )

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._write("chain_end", run_id, parent_run_id, outputs=_safe(outputs))

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._write("chain_error", run_id, parent_run_id, error=str(error))


def create_trace_logger(thread_id):
    os.makedirs(TRACES_DIR, exist_ok=True)
    trace_path = os.path.join(TRACES_DIR, f"trace_{thread_id}.jsonl")
    return JSONLTraceLogger(trace_path, thread_id=thread_id), trace_path
