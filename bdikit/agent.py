import os
import re
import sys
import ast
import json
import time
import uuid
import asyncio
import shlex
import argparse
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage

# Reuse the raw JSONL logger from chatbot_tracing.py (next to this file first,
# then the installed bdikit package).
try:
    from chatbot_tracing import JSONLTraceLogger, TRACES_DIR
except ImportError:
    from bdikit.chatbot_tracing import JSONLTraceLogger, TRACES_DIR


# ---------------------------------------------------------------------------
# System prompt: read it from chatbot.py so both stay in sync.
# (chatbot.py relaunches Streamlit on import, so we parse it instead.)
# ---------------------------------------------------------------------------
def load_system_prompt():
    candidates = [os.path.join(HERE, "chatbot.py")]
    try:
        import bdikit

        candidates.append(os.path.join(os.path.dirname(bdikit.__file__), "chatbot.py"))
    except ImportError:
        pass
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets
                )
                and isinstance(node.value, ast.Constant)
            ):
                return node.value.value, path
    raise RuntimeError("Could not find the `prompt` string in chatbot.py")


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
OPENAI_REASONING = re.compile(r"^(openai/)?(o\d|gpt-[5-9])", re.IGNORECASE)


def build_llm(model, effort):
    if OPENAI_REASONING.match(model):
        # Same idea as Roque's script: Responses API + detailed reasoning summary.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            sys.exit(
                f"langchain-openai is not installed for this Python ({sys.executable}).\n"
                f"Fix: {sys.executable} -m pip install langchain-openai"
            )

        return ChatOpenAI(
            model=model.split("/", 1)[-1] if model.startswith("openai/") else model,
            use_responses_api=True,
            reasoning={"effort": effort, "summary": "detailed"},
            max_retries=1,
            timeout=300,
        )

    # Any other model goes through LiteLLM, same as chatbot.py: Claude
    # (anthropic/...), Gemini (gemini/...), DeepInfra, Portkey, local models...
    from langchain_litellm import ChatLiteLLM

    kwargs = {}
    if re.search(r".+/@.+", model):  # Portkey, same as chatbot.py
        kwargs = {
            "api_base": os.getenv("PORTKEY_API_BASE"),
            "extra_headers": {"x-portkey-api-key": os.getenv("PORTKEY_API_KEY")},
        }
    model_kwargs = {}
    if effort:
        # Ask for reasoning when the provider supports it; LiteLLM drops the
        # parameter for models that don't. Reasoning comes back as
        # reasoning_content / thinking blocks, which the tracer captures.
        model_kwargs = {"reasoning_effort": effort, "drop_params": True}
    return ChatLiteLLM(
        model=model,
        streaming=True,
        max_retries=1,
        timeout=300,
        model_kwargs=model_kwargs,
        **kwargs,
    )


# Extra instructions appended to chatbot.py's prompt (turn off with --plain).
TRACE_INSTRUCTIONS = """

========================
EXPLAIN EACH STEP (for the trace)
========================
Every tool has a required `why` argument. Fill it with one short sentence saying
why you are calling this tool and why you chose this method. Example:
why: "Gender and Race look like exact name matches, so a cheap non-LLM method is enough."

========================
CHOOSING METHODS
========================
You may use ANY bdi-kit method, not only the defaults. Always set the `method`
argument explicitly. Pick the method that gives the best answer for the least cost.

Schema matching (match_schema):
- No LLM, cheap: coma, cupid, similarity_flooding, distribution_based, jaccard_distance,
  two_phase, max_val_sim, magneto_zs_bp, magneto_ft_bp
- Uses an LLM, more accurate on hard columns, costs tokens: magneto_zs_llm, magneto_ft_llm, llm
Ranking alternatives (rank_schema_matches): max_val_sim, magneto_zs_bp, magneto_ft_bp,
  magneto_zs_llm, magneto_ft_llm, llm
Value matching (match_values): edit_distance, tfidf, embedding (no LLM);
  llm, llm_numeric (use an LLM)
Ranking values (rank_value_matches): tfidf, embedding, llm

Call get_available_schema_matching_algorithms / get_available_value_matching_algorithms
if you need the descriptions. If a method fails or gives weak results, say so and try another.
When you need alternatives for several columns, pass them all in ONE call
(the `attributes` argument takes a list) instead of one call per column.
When you need to look at several columns (preview_domain) or make several fixes
(update_schema_matching / update_value_matching), request them ALL in the same step
as parallel tool calls, not one per step. Only check columns that are flagged
(low score, odd name/value fit); do not re-check confident matches.

========================
WORKFLOW: SCHEMA -> VALUES -> HARMONIZE
========================
Work in three stages and STOP after each one so the user can review:
1. Schema matching: run match_schema, review, show the table, then ask the user
   to confirm or change matches before value matching.
2. Value matching: when the user says go ahead, run match_values for the matched
   columns (all of them - this is biomedical data, do not skip any). Columns whose
   target has no fixed list of values (numbers like BMI, Age) are kept as-is
   automatically; tell the user which ones. Show the results and ask the user to
   confirm or change them.
3. Harmonization: when the user says go ahead, ask for the output folder if they
   did not give one, then call materialize_mapping and report the saved file path.

CHANGES ARE INCREMENTAL. Never redo work that is not affected by the user's change:
- "map X to Y": call update_schema_matching for X only. If X already had value
  matches, re-run match_values for the pair (X, Y) only.
- "redo X with method M": call rank_schema_matches with attributes=[X] and method M,
  then update_schema_matching for X with the best candidate. Do NOT re-run match_schema.
- "value v of X should be w": call update_value_matching for that value only.
- "redo the values of X with method M": call match_values with attribute_matches
  set to only that column pair.
- Use the results you already have instead of recomputing.
If a message starts with "[Changes the user already made]", those changes are
already applied; take them into account and do not redo them.
"""


WHY_FIELD = {
    "type": "string",
    "description": "Required. One short sentence: why you are calling this tool now, "
    "and why you chose this method / these arguments.",
}


MAX_ROWS_FOR_LLM = 300  # bigger tool results are summarized before the model sees them

_ALLOWED_CACHE = {}


def target_allowed_values(target, attribute):
    """Allowed values of a target column: a list (empty = free values like numbers,
    e.g. GDC bmi / age_at_index), or None if unknown."""
    key = (target, attribute)
    if key in _ALLOWED_CACHE:
        return _ALLOWED_CACHE[key]
    allowed = None
    try:
        if target and os.path.exists(str(target)):
            import pandas as pd

            col = pd.read_csv(target, usecols=[attribute])[attribute]
            allowed = (
                []
                if pd.api.types.is_numeric_dtype(col)
                else list(col.dropna().unique())
            )
        else:
            from bdikit.standards.standard_factory import Standards

            meta = Standards.get_standard(target or "gdc").get_attribute_metadata(
                [attribute]
            )
            allowed = list(meta[attribute]["value_names"])
    except Exception:
        allowed = None
    _ALLOWED_CACHE[key] = allowed
    return allowed


def split_value_pairs(pairs, target):
    """Pairs whose target has a fixed list of values (match them) vs free values (keep as-is)."""
    to_match, keep_as_is = [], []
    for p in pairs:
        allowed = target_allowed_values(target, p["target_attribute"])
        (keep_as_is if allowed == [] else to_match).append(p)
    return to_match, keep_as_is


def compact_for_llm(tool_name, rows):
    """Short summary of a big tool result for the model (full rows stay in the trace)."""
    if rows and "source_value" in rows[0]:
        groups = {}
        for r in rows:
            groups.setdefault(
                (r.get("source_attribute"), r.get("target_attribute")), []
            ).append(r)
        lines = [
            f"{tool_name} returned {len(rows)} value matches for {len(groups)} column pairs. "
            f"This is a summary per column (n = values, unmapped = no target value, low = score <= {LOW}). "
            "All matches are stored on the server; ask the user before changing them."
        ]
        for (src, tgt), rs in groups.items():
            unmapped = sum(1 for r in rs if _missing(r.get("target_value")))
            low = sum(
                1
                for r in rs
                if not _missing(r.get("target_value"))
                and _f(r.get("similarity")) <= LOW
            )
            ex = "; ".join(
                f"{r.get('source_value')!s:.30} -> {r.get('target_value')!s:.30}"
                for r in rs[:3]
            )
            lines.append(
                f"- {src} -> {tgt}: n={len(rs)}, unmapped={unmapped}, low={low}. e.g. {ex}"
            )
        return "\n".join(lines)
    head = json.dumps(rows[:MAX_ROWS_FOR_LLM], default=str)
    return f"{tool_name} returned {len(rows)} rows; first {MAX_ROWS_FOR_LLM} shown:\n{head}"


def agent_tool(tool, add_why=True, current_pairs=None):
    """Copy of an MCP tool for the agent:
    - adds a required `why` argument so the model states its reason on every call
      (works on any model); it is removed before the call reaches the MCP server
    - big results (e.g. 12,000 value matches) are summarized for the model so they
      don't overflow its context; the full result goes to the trace as the artifact.
    The MCP server itself is unchanged."""
    from typing import Annotated
    from langchain_core.tools import InjectedToolArg

    schema = (
        json.loads(json.dumps(tool.args_schema))
        if isinstance(tool.args_schema, dict)
        else None
    )
    if schema is None or tool.coroutine is None:
        return tool
    if add_why:
        schema.setdefault("properties", {})["why"] = WHY_FIELD
        schema["required"] = ["why"] + [
            r for r in schema.get("required", []) if r != "why"
        ]
    original = tool.coroutine

    async def call(
        runtime: Annotated[object | None, InjectedToolArg()] = None, **arguments
    ):
        arguments.pop("why", None)
        # drop empty settings (e.g. method_args: null), so the server uses its defaults
        # instead of failing ("'NoneType' object has no attribute 'update'")
        arguments = {k: v for k, v in arguments.items() if v is not None}
        extra = {}
        note = ""
        if tool.name == "match_values":
            # columns whose target has no fixed list of values (numbers like BMI, Age)
            # keep their values as-is instead of being "matched" to nothing
            pairs = arguments.get("attribute_matches") or (
                current_pairs() if current_pairs else None
            )
            if pairs:
                to_match, keep = split_value_pairs(
                    pairs, arguments.get("target_dataset_path")
                )
                if keep:
                    extra["bdi_kept_as_is"] = [
                        f"{p['source_attribute']} -> {p['target_attribute']}"
                        for p in keep
                    ]
                    note = (
                        f"Not value-matched (target has no fixed list of values, values kept as-is): "
                        f"{', '.join(extra['bdi_kept_as_is'])}\n"
                    )
                if not to_match:
                    return [
                        {
                            "type": "text",
                            "text": note + "No columns left to value-match.",
                        }
                    ], extra
                arguments["attribute_matches"] = to_match
        content, artifact = await original(runtime=runtime, **arguments)
        rows = parse_tool_output(content)
        if (
            isinstance(rows, list)
            and len(rows) > MAX_ROWS_FOR_LLM
            and isinstance(rows[0], dict)
        ):
            summary = compact_for_llm(tool.name, rows)
            return (
                [{"type": "text", "text": note + summary}],
                {**extra, "bdi_full_rows": rows, "mcp_artifact": artifact},
            )
        if note:
            text = (
                content if isinstance(content, str) else json.dumps(rows, default=str)
            )
            return [{"type": "text", "text": note + text}], {
                **extra,
                "bdi_full_rows": rows if isinstance(rows, list) else [],
                "mcp_artifact": artifact,
            }
        return content, artifact

    return tool.model_copy(update={"args_schema": schema, "coroutine": call})


def _flatten_content(content):
    if not isinstance(content, list):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            if isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts)


def sanitize_messages_for_llm(state):
    """Same as chatbot.py, plus: drop old reasoning items before replaying
    history (they are already saved in the trace and some APIs reject them)."""
    out = []
    for m in state["messages"]:
        if isinstance(m, (AIMessage, ToolMessage)):
            update = {}
            if isinstance(m.content, list):
                update["content"] = _flatten_content(m.content)
            extra = getattr(m, "additional_kwargs", {}) or {}
            if "reasoning" in extra or "reasoning_content" in extra:
                update["additional_kwargs"] = {
                    k: v
                    for k, v in extra.items()
                    if k not in ("reasoning", "reasoning_content")
                }
            if update:
                m = m.model_copy(update=update)
        out.append(m)
    return {"llm_input_messages": out}


# ---------------------------------------------------------------------------
# Helpers to pull facts out of callback payloads
# ---------------------------------------------------------------------------
def extract_reasoning(msg):
    """Reasoning text from an AIMessage, whatever provider produced it."""
    texts = []
    extra = getattr(msg, "additional_kwargs", {}) or {}

    r = extra.get("reasoning")  # OpenAI Responses API (langchain-openai)
    if isinstance(r, dict):
        for s in r.get("summary") or []:
            if isinstance(s, dict) and s.get("text"):
                texts.append(s["text"])
    if isinstance(extra.get("reasoning_content"), str):  # LiteLLM / gpt-oss
        texts.append(extra["reasoning_content"])

    content = getattr(msg, "content", None)
    if isinstance(content, list):  # content blocks (newer langchain output formats)
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "reasoning":
                if isinstance(b.get("reasoning"), str):
                    texts.append(b["reasoning"])
                for s in b.get("summary") or []:
                    if isinstance(s, dict) and s.get("text"):
                        texts.append(s["text"])
            elif b.get("type") == "thinking" and isinstance(b.get("thinking"), str):
                texts.append(b["thinking"])

    seen, unique = set(), []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return "\n\n".join(unique)


def extract_usage(msg, llm_output=None):
    u = getattr(msg, "usage_metadata", None) or {}
    inp, out = u.get("input_tokens"), u.get("output_tokens")
    reasoning = (u.get("output_token_details") or {}).get("reasoning")
    if inp is None and llm_output:
        tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
        inp = tu.get("prompt_tokens") or tu.get("input_tokens")
        out = tu.get("completion_tokens") or tu.get("output_tokens")
    return inp or 0, out or 0, reasoning or 0


def parse_tool_output(output):
    """MCP tool output -> python object (list/dict) when possible."""
    content = getattr(output, "content", output)
    if isinstance(content, list):
        items = []
        for b in content:
            text = b.get("text") if isinstance(b, dict) else b
            items.append(_try_json(text))
        flat = []
        for it in items:
            flat.extend(it if isinstance(it, list) else [it])
        return flat[0] if len(flat) == 1 and isinstance(flat[0], dict) else flat
    return _try_json(content)


def _try_json(text):
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text


def short(value, n=300):
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def fmt_sim(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "?"


LOW = 0.5  # same threshold the system prompt uses to flag matches

TOOL_DEFAULT_METHOD = {
    "match_schema": "magneto_ft_llm",
    "rank_schema_matches": "magneto_ft_llm",
    "match_values": "llm",
    "rank_value_matches": "llm",
}

TOOL_PURPOSE = {
    "match_schema": "find which target column each source column maps to",
    "rank_schema_matches": "list alternative target columns for source columns",
    "match_values": "map source values to the target's allowed values",
    "rank_value_matches": "list alternative target values for source values",
    "preview_domain": "look at a column's description and example values",
    "materialize_mapping": "apply the mappings and write the harmonized CSV",
    "update_schema_matching": "manually change a column mapping",
    "update_value_matching": "manually change a value mapping",
    "get_available_schema_matching_algorithms": "list schema matching methods",
    "get_available_value_matching_algorithms": "list value matching methods",
}


def describe_tool_result(tool, result):
    """Plain-English lines for a tool result."""
    lines = []
    rows = result if isinstance(result, list) else None

    if tool == "match_schema" and rows is not None:
        matched = [r for r in rows if not _missing(r.get("target_attribute"))]
        unmatched = [
            r.get("source_attribute")
            for r in rows
            if _missing(r.get("target_attribute"))
        ]
        low = [r for r in matched if _f(r.get("similarity")) <= LOW]
        lines.append(
            f"Found {len(matched)} column matches ({len(low)} low-confidence, score <= {LOW}):"
        )
        for r in matched:
            flag = "  LOW" if _f(r.get("similarity")) <= LOW else ""
            lines.append(
                f"  - {r.get('source_attribute')} -> {r.get('target_attribute')} ({fmt_sim(r.get('similarity'))}){flag}"
            )
        if unmatched:
            lines.append(
                f"No match found for {len(unmatched)} column(s): {', '.join(map(str, unmatched))}"
            )

    elif tool == "rank_schema_matches" and rows is not None:
        by_src = {}
        for r in rows:
            by_src.setdefault(r.get("source_attribute"), []).append(r)
        lines.append(f"Got ranked candidates for {len(by_src)} source column(s):")
        for src, cands in by_src.items():
            top = ", ".join(
                f"{c.get('target_attribute')} ({fmt_sim(c.get('similarity'))})"
                for c in cands[:5]
            )
            lines.append(f"  - {src}: {top}")

    elif tool in ("match_values", "rank_value_matches") and rows is not None:
        low = [r for r in rows if _f(r.get("similarity")) <= LOW]
        pairs = {(r.get("source_attribute"), r.get("target_attribute")) for r in rows}
        lines.append(
            f"Got {len(rows)} value matches across {len(pairs)} column pair(s) ({len(low)} low-confidence):"
        )
        for r in rows[:25]:
            flag = "  LOW" if _f(r.get("similarity")) <= LOW else ""
            lines.append(
                f"  - [{r.get('source_attribute')}] '{r.get('source_value')}' -> '{r.get('target_value')}' ({fmt_sim(r.get('similarity'))}){flag}"
            )
        if len(rows) > 25:
            lines.append(f"  ... and {len(rows) - 25} more (see JSONL trace)")

    elif tool == "preview_domain":
        lines.append(f"Domain preview: {short(result, 400)}")

    elif tool in ("update_schema_matching", "update_value_matching"):
        pass  # the change itself is shown in "What changed in the mappings"

    elif tool == "materialize_mapping" and isinstance(result, dict):
        n = len(result.get("data") or [])
        lines.append(f"{result.get('message', '')} ({n} rows)")

    else:
        lines.append(f"Result: {short(result, 400)}")
    return lines


def _f(x):
    try:
        return float(x)
    except Exception:
        return 1.0


def _missing(x):
    return (
        x is None
        or (isinstance(x, float) and x != x)
        or str(x).lower() in ("nan", "none", "")
    )


# ---------------------------------------------------------------------------
# Plain-English trace
# ---------------------------------------------------------------------------
class NarrativeTracer(BaseCallbackHandler):
    """Turns LangChain callback events into a readable step-by-step trace."""

    def __init__(self, md_path, agent_model, mcp_model):
        self.md_path = md_path
        self.agent_model = agent_model
        self.mcp_model = mcp_model
        self.step = 0
        self.turn = 0
        self.llm_start = {}
        self.tool_start = {}
        self.turn_tools = []
        self.turn_log = []
        self.candidates = {}  # source column -> {target: {score, method, tool}}
        self.manual = False
        self.mirror_schema = []  # copy of the server's schema matches
        self.mirror_values = []  # copy of the server's value matches
        self.source_path = None  # last source dataset the tools were run on
        self.target_path = "gdc"  # last target schema
        self.value_method = None  # last value-matching method used
        self.totals = self._new_totals()
        os.makedirs(os.path.dirname(md_path), exist_ok=True)

    @staticmethod
    def _new_totals():
        return {
            "llm_calls": 0,
            "in": 0,
            "out": 0,
            "reasoning": 0,
            "tools": {},
            "methods": {},
            "llm_s": 0.0,
            "tool_s": 0.0,
        }

    def _w(self, *lines):
        with open(self.md_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    # turn boundaries are called from run_turn()
    def begin_turn(self, user_text, manual=False):
        self._turn_offset = (
            os.path.getsize(self.md_path) if os.path.exists(self.md_path) else 0
        )
        self.turn += 1
        self.step = 0
        self.totals = self._new_totals()
        self.turn_tools = []  # (tool name, args) of successful calls this turn
        self.turn_log = []  # structured steps of this turn (input for the narrative)
        self.manual = manual
        if manual:
            self._w(
                "",
                f"## Turn {self.turn} (manual command - no LLM)",
                "",
                f"**User ran:** `{user_text}`",
                "",
            )
        else:
            self._w("", f"## Turn {self.turn}", "", f"**User asked:** {user_text}", "")

    def _mirror(self, name, result):
        """Keep a copy of what the MCP server holds, built only from tool outputs
        (the server is not changed): match_schema / update_schema_matching return
        the schema table; match_values returns the rows that replace that column's old ones;
        update_value_matching returns the full value table."""
        rows = (
            [r for r in result if isinstance(r, dict)]
            if isinstance(result, list)
            else None
        )
        if rows is None:
            return
        if name in ("match_schema", "update_schema_matching"):
            self.mirror_schema = rows
        elif name == "match_values":
            # the server replaces the rows of re-run column pairs
            pairs = {
                (r.get("source_attribute"), r.get("target_attribute")) for r in rows
            }
            kept = [
                r
                for r in self.mirror_values
                if (r.get("source_attribute"), r.get("target_attribute")) not in pairs
            ]
            self.mirror_values = kept + rows
        elif name == "update_value_matching":
            self.mirror_values = rows

    def snapshot(self):
        return {
            "schema_matches": self.mirror_schema,
            "value_matches": self.mirror_values,
        }

    def write_changes(self, change_lines):
        self._w("### What changed in the mappings", "")
        self._w(*(change_lines or ["- Nothing changed."]), "")

    def end_turn(self, final_text):
        t = self.totals
        tools = ", ".join(f"{k} x{v}" for k, v in t["tools"].items()) or "none"
        methods = ", ".join(f"{k} x{v}" for k, v in t["methods"].items()) or "none"
        self._w(
            "### Final answer",
            "",
            final_text.strip() or "(empty)",
            "",
            "### Summary of this turn",
            "",
            f"- LLM calls: {t['llm_calls']} ({t['llm_s']:.1f}s)",
            f"- Agent tokens: {t['in']} in / {t['out']} out (of which {t['reasoning']} reasoning)",
            f"- Tool calls: {tools} ({t['tool_s']:.1f}s), failed: {t.get('tool_errors', 0)}",
            f"- Matching methods used: {methods}",
            "- Note: methods with 'llm' in the name call an LLM inside the MCP server; "
            "those tokens are NOT counted above.",
            "",
        )

    def turn_text(self):
        """The plain-English trace of the turn that just finished."""
        with open(self.md_path) as f:
            f.seek(getattr(self, "_turn_offset", 0))
            return f.read()

    # --- LLM events -------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.llm_start[run_id] = time.time()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self.llm_start[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs):
        dt = time.time() - self.llm_start.pop(run_id, time.time())
        try:
            msg = response.generations[0][0].message
        except Exception:
            return
        inp, out, rsn = extract_usage(msg, getattr(response, "llm_output", None))
        t = self.totals
        t["llm_calls"] += 1
        t["in"] += inp
        t["out"] += out
        t["reasoning"] += rsn
        t["llm_s"] += dt

        self.step += 1
        lines = [
            f"### Step {self.step}: LLM thinks ({self.agent_model}, {dt:.1f}s, {inp} in / {out} out tokens"
            + (f", {rsn} reasoning" if rsn else "")
            + ")",
            "",
        ]
        calls = getattr(msg, "tool_calls", None) or []
        stated = (_flatten_content(msg.content) or "").strip() if calls else ""
        whys = [
            c.get("args", {}).get("why") for c in calls if c.get("args", {}).get("why")
        ]
        if whys and not stated:
            stated = " / ".join(whys)
        calls = [
            {
                **c,
                "args": {k: v for k, v in (c.get("args") or {}).items() if k != "why"},
            }
            for c in calls
        ]
        reasoning = extract_reasoning(msg)

        if stated:  # what the agent wrote before the tool call ("Why: ...")
            lines += [f"**Why (agent's own words):** {stated}", ""]
        if reasoning:  # hidden reasoning the provider returned (summary / thinking)
            lines += ["**Internal reasoning (from the model):**", ""]
            lines += ["> " + ln if ln.strip() else ">" for ln in reasoning.splitlines()]
            lines.append("")
        if not stated and not reasoning:
            lines += ["_No reasoning returned for this step._", ""]

        if calls:
            for c in calls:
                lines.append(
                    f"**Decided to call:** `{c.get('name')}` with {short(c.get('args', {}), 300)}"
                )
        else:
            lines.append("**Decided:** answer the user directly (no tool call).")
        lines.append("")
        self._w(*lines)
        self.turn_log.append(
            {
                "step": self.step,
                "kind": "agent thinking",
                "agent_why": stated,
                "model_reasoning": reasoning[:2000],
                "decided": [
                    {"tool": c.get("name"), "args": _short_args(c.get("args", {}))}
                    for c in calls
                ]
                or "answer the user",
            }
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.llm_start.pop(run_id, None)
        self._w(f"**LLM error:** {short(str(error), 500)}", "")

    # --- Tool events ------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        args = inputs if isinstance(inputs, dict) else _try_json(input_str)
        args = (
            {k: v for k, v in args.items() if k != "why"}
            if isinstance(args, dict)
            else {}
        )
        self.tool_start[run_id] = (name, args, time.time())

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, args, t0 = self.tool_start.pop(run_id, ("tool", {}, time.time()))
        dt = time.time() - t0
        self.totals["tool_s"] += dt
        self.totals["tools"][name] = self.totals["tools"].get(name, 0) + 1

        self.step += 1
        lines = [f"### Step {self.step}: Tool `{name}` ({dt:.1f}s)", ""]
        if name in TOOL_PURPOSE:
            lines.append(f"- Purpose: {TOOL_PURPOSE[name]}")
        if name == "update_schema_matching":
            lines.append(
                f"- Set: **{args.get('source_attribute')}** -> {args.get('new_target_attribute')}"
            )
        if name == "update_value_matching":
            lines.append(
                f"- Set: [{args.get('source_attribute')}] '{args.get('source_value')}' -> "
                f"'{args.get('new_target_value')}'"
            )
        if name in TOOL_DEFAULT_METHOD:
            method = args.get("method") or TOOL_DEFAULT_METHOD[name]
            who = "the user" if self.manual else "the agent"
            how = (
                f"chosen by {who}"
                if args.get("method")
                else f"default - {who} did not choose"
            )
            self.totals["methods"][method] = self.totals["methods"].get(method, 0) + 1
            uses_llm = "llm" in method
            lines.append(
                f"- Method: **{method}** ({how})"
                + (
                    f"; uses LLM `{self.mcp_model}` internally"
                    if uses_llm
                    else "; no LLM call"
                )
            )
        for key in (
            "source_dataset_path",
            "dataset_path",
            "target_dataset_path",
            "attribute",
            "attributes",
            "top_k",
        ):
            if key in args and args[key] not in (None, ""):
                val = args[key]
                if isinstance(val, str) and key.endswith("_path") and os.sep in val:
                    val = os.path.basename(val)
                lines.append(f"- {key.replace('_', ' ')}: {short(val, 200)}")
        if args.get("attribute_matches"):
            lines.append(
                f"- column pairs given: {short(args['attribute_matches'], 300)}"
            )
        result = parse_tool_output(output)
        artifact = getattr(output, "artifact", None)
        if isinstance(artifact, dict) and artifact.get("bdi_kept_as_is"):
            kept = artifact["bdi_kept_as_is"]
            lines.append(
                f"- Kept as-is, no value matching ({len(kept)} columns with free values like numbers): "
                + short(", ".join(kept), 600)
            )
        if isinstance(artifact, dict) and "bdi_full_rows" in artifact:
            result = artifact[
                "bdi_full_rows"
            ]  # the model got a summary; trace the full result
            if len(result) > MAX_ROWS_FOR_LLM:
                lines.append(
                    f"- Result too big for the model ({len(result)} rows): it received a per-column summary"
                )
        is_error = getattr(output, "status", None) == "error" or (
            # direct tool calls (quick commands) return the error as plain text
            "Error executing tool"
            in short(result, 200)
        )
        if is_error:
            # MCP adapter returns tool failures as a normal message with status="error"
            self.totals["tool_errors"] = self.totals.get("tool_errors", 0) + 1
            lines[0] = f"### Step {self.step}: Tool `{name}` FAILED ({dt:.1f}s)"
            lines += ["", f"**Error:** {short(result, 500)}", ""]
        else:
            self.turn_tools.append((name, args))
            if args.get("source_dataset_path"):
                self.source_path = args["source_dataset_path"]
            if args.get("target_dataset_path"):
                self.target_path = args["target_dataset_path"]
            if name in ("match_values", "rank_value_matches"):
                self.value_method = args.get("method") or TOOL_DEFAULT_METHOD[name]
            self._mirror(name, result)
            self._remember_candidates(name, args, result)
            try:
                lines += ["", *describe_tool_result(name, result), ""]
            except Exception:
                lines += ["", f"Result: {short(result, 400)}", ""]
        self._w(*lines)
        self.turn_log.append(
            {
                "step": self.step,
                "kind": "tool call",
                "tool": name,
                "tool_purpose": TOOL_PURPOSE.get(name, ""),
                "method": (
                    args.get("method") or TOOL_DEFAULT_METHOD[name]
                    if name in TOOL_DEFAULT_METHOD
                    else None
                ),
                "method_chosen_by": (
                    ("user" if self.manual else "agent")
                    if args.get("method")
                    else "default"
                ),
                "inputs": _short_args(args),
                "failed": is_error,
                "result": [l for l in lines if l and not l.startswith("###")][:25],
            }
        )

    def _remember_candidates(self, name, args, result):
        """Every target bdi-kit suggested per source column (for runner-ups)."""
        if name not in ("match_schema", "rank_schema_matches") or not isinstance(
            result, list
        ):
            return
        method = args.get("method") or TOOL_DEFAULT_METHOD[name]
        for r in result:
            if not isinstance(r, dict) or _missing(r.get("target_attribute")):
                continue
            seen = self.candidates.setdefault(r.get("source_attribute"), {})
            tgt = r.get("target_attribute")
            score = r.get("similarity")
            if tgt not in seen or _f(score) > _f(seen[tgt]["score"]):
                seen[tgt] = {"score": score, "method": method, "tool": name}

    def on_tool_error(self, error, *, run_id, **kwargs):
        name, args, _ = self.tool_start.pop(run_id, ("tool", {}, time.time()))
        self.totals["tool_errors"] = self.totals.get("tool_errors", 0) + 1
        self.step += 1
        self.turn_log.append(
            {
                "step": self.step,
                "kind": "tool call",
                "tool": name,
                "inputs": _short_args(args),
                "failed": True,
                "result": [f"Error: {short(str(error), 300)}"],
            }
        )
        self._w(
            f"### Step {self.step}: Tool `{name}` FAILED",
            "",
            f"- Args: {short(args, 300)}",
            f"- Error: {short(str(error), 500)}",
            "",
        )


def _free_base(folder, base):
    """base, or base_2, base_3 ... if trace files with that name already exist."""
    candidate, n = base, 1
    while any(
        os.path.exists(os.path.join(folder, candidate + ext))
        for ext in (".md", ".jsonl", "_state.json")
    ):
        n += 1
        candidate = f"{base}_{n}"
    return candidate


def _short_args(args):
    """Tool arguments for the narrative: file names instead of full paths, no long lists."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and k.endswith("_path") and os.sep in v:
            v = os.path.basename(v)
        elif isinstance(v, list) and len(v) > 10:
            v = v[:10] + [f"... +{len(v) - 10} more"]
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Narrative: a plain-English story of each turn for people who don't know the
# chatbot. Facts come only from the trace (steps, scores, candidates) plus cheap
# checks on the data (no LLM); a narrator LLM turns those facts into the story.
# ---------------------------------------------------------------------------
# How each bdi-kit method works, in plain words (based on the bdi-kit docstrings)
METHOD_GUIDE = {
    # schema matching
    "magneto_ft_llm": "Magneto, 2 steps: (1) a small AI language model, fine-tuned on biomedical "
    "columns, turns each column's name and example values into a 'meaning fingerprint' and finds "
    "the target columns with the closest fingerprints; (2) a large LLM re-reads those few "
    "candidates and puts them in order. Score 0-1 = how confident it is.",
    "magneto_zs_llm": "Magneto, 2 steps: (1) a small general-purpose AI language model (not "
    "fine-tuned) finds target columns with a similar meaning; (2) a large LLM re-orders them.",
    "magneto_ft_bp": "Magneto, 2 steps: (1) a small fine-tuned AI language model finds target "
    "columns with a similar meaning; (2) a math algorithm (bipartite matching) picks the best "
    "one-to-one pairs. No large LLM, so it is fast and free.",
    "magneto_zs_bp": "Magneto, 2 steps: (1) a small general-purpose AI language model finds "
    "similar target columns; (2) a math algorithm (bipartite matching) picks one-to-one pairs. No LLM.",
    "llm": "A large LLM reads the source column (name and example values) and the target columns "
    "and picks the best match directly.",
    "coma": "COMA combines several simple checks on the column names and the structure of the "
    "schemas (no AI).",
    "cupid": "Cupid compares column names by their words and meaning plus the structure of the "
    "schemas (no AI).",
    "similarity_flooding": "Similarity Flooding turns both schemas into graphs and spreads "
    "similarity between neighbouring columns until it settles (no AI).",
    "distribution_based": "Compares how the values in each column are distributed "
    "(Earth Mover's Distance) (no AI).",
    "jaccard_distance": "Jaccard compares how many values two columns share, treating values with "
    "very small spelling differences as the same (no AI).",
    "two_phase": "Two phases: a ranking method first keeps the most likely target columns, then a "
    "second matcher picks the final match among them.",
    "max_val_sim": "First keeps the most likely target columns (with a fast ranking method), then "
    "picks the one whose values are most similar to the source values.",
    # value matching
    "tfidf": "TF-IDF compares the spelling of values by the groups of letters they share "
    "(e.g. 'Male' vs 'male'); no AI.",
    "embedding": "Compares the meaning of values using a small AI language model "
    "(e.g. 'heart disorder' is close to 'Cardiovascular Disorder').",
    "edit_distance": "Counts how many letter changes turn one value into another; fewest wins (no AI).",
    "llm_numeric": "A large LLM writes a small formula to convert numbers from the source format "
    "to the target format (e.g. units).",
}
VALUE_LLM_GUIDE = "A large LLM reads each source value and the list of allowed target values and picks the best one."

NARRATOR_PROMPT = """You explain what an AI data-harmonization agent did, for a reader who has never
used it and knows nothing about data matching or AI.

Rules:
- Use ONLY the facts in the JSON. Never invent numbers, columns, scores or reasons. If something is
  missing, write "not recorded". Point to the step it came from, like (Step 3).
- Only call something done if it ran. In "steps", a decided tool call with "executed": false was only
  planned and never ran - say "planned but not done". The "decisions" list is the final result.
- If "turn_status" is not "completed", say at the top, in one sentence, that the run stopped early
  and the user did not get a final answer for this request.
- Explain each technical word once, the first time (schema matching, value matching, GDC, score).
- Short, plain sentences. No hype. Do not offer further help or next steps. Write nothing outside
  the two parts below.

#### What happened
3-7 numbered points that follow the steps, like a worked example: which tool and why it fit the
request; which method and why (the agent's "why" and the model's reasoning; say if it was just the
default); how that method works, step by step, applied to THIS data (use method_guide), e.g.
"it read the name 'Age' and values like 54 and 72, compared their meaning with the GDC columns, and
the closest was age_at_index with a score of 0.95 out of 1"; and what came out. Group repeated
steps (e.g. "checked 9 columns (Steps 16-24)") instead of listing each.

#### Why each match
One bullet per item in "decisions". Write ONLY the version its "detail" asks for - never both:
- "full" (AI-corrected or changed by the user), at most ~120 words, three labelled parts:
  **Why this method**, **Why this column** (evidence: what the target column means, shared name
  words, example values and whether they fit the allowed values, the score), **Other options and why
  they lost** (candidates and scores; if there are none, say "no other options were recorded").
  If "ai_own_pick" is true, end with: "bdi-kit never suggested this column; the AI picked it from
  its own knowledge, so please double-check it."
- "medium": ONE sentence of at most ~40 words: chosen column, the key evidence, the score.
For value-mapping items: the method and why, 1-3 examples, and how many values were not mapped or
kept as-is and why (same length rules).
"""


class Evidence:
    """Cheap checks on the data (no LLM) that explain why a target column fits."""

    def __init__(self):
        self._frames = {}

    def _frame(self, path):
        if path not in self._frames:
            import pandas as pd

            try:
                self._frames[path] = pd.read_csv(path)
            except Exception:
                self._frames[path] = None
        return self._frames[path]

    @staticmethod
    def _words(name):
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
        return {w for w in re.split(r"[^a-zA-Z0-9]+", name.lower()) if len(w) > 1}

    def _target_info(self, target, attribute):
        try:
            if target and os.path.exists(str(target)):
                df = self._frame(target)
                col = df[attribute]
                return {
                    "meaning": "not recorded (target is a CSV file)",
                    "allowed": list(col.dropna().astype(str).unique()[:15]),
                }
            from bdikit.standards.standard_factory import Standards

            meta = Standards.get_standard(target or "gdc").get_attribute_metadata(
                [attribute]
            )[attribute]
            allowed = meta.get("value_names")
            if isinstance(allowed, str):
                allowed = _try_json(allowed) if allowed.startswith("[") else [allowed]
            return {
                "meaning": str(meta.get("attribute_description", ""))[:300],
                "allowed": list(allowed or [])[:15],
            }
        except Exception:
            return {"meaning": "not recorded", "allowed": None}

    def for_pair(self, source_path, target, src, tgt):
        ev = {"target_column": tgt}
        df = self._frame(source_path) if source_path else None
        values = []
        if df is not None and src in df.columns:
            col = df[src]
            values = [str(v) for v in col.dropna().unique()[:8]]
            ev["source_example_values"] = values
            ev["source_kind"] = (
                "numbers"
                if str(col.dtype).startswith(("int", "float"))
                else "text / categories"
            )
            ev["source_distinct_values"] = int(col.nunique())
        info = self._target_info(target, tgt)
        ev["target_meaning"] = info["meaning"]
        if info["allowed"] is not None:
            ev["target_allowed_values"] = (
                info["allowed"] or "free values (any number/text)"
            )
            if info["allowed"] and values:
                allowed = {str(a).lower() for a in info["allowed"]}
                hits = [v for v in values if v.lower() in allowed]
                ev["example_values_found_in_allowed_list"] = (
                    f"{len(hits)} of {len(values)}"
                )
        ev["shared_name_words"] = sorted(self._words(src) & self._words(tgt)) or "none"
        return ev


def mark_executed(turn_log):
    """Copy of the steps where every tool the agent decided to call says whether it
    actually ran (a later tool step with that name) or was only planned (e.g. the
    run stopped first)."""
    steps = json.loads(json.dumps(turn_log, default=str))
    for i, s in enumerate(steps):
        if s.get("kind") != "agent thinking" or not isinstance(s.get("decided"), list):
            continue
        # tool steps that follow this decision, up to the next agent-thinking step
        ran = []
        for later in steps[i + 1 :]:
            if later.get("kind") == "agent thinking":
                break
            ran.append(later.get("tool"))
        for d in s["decided"]:
            if d.get("tool") in ran:
                ran.remove(d["tool"])
                d["executed"] = True
            else:
                d["executed"] = False
                d["note"] = "planned only - this call never ran"
    return steps


def build_turn_facts(narrator, state, evidence, request, s_changed, v_changed, final):
    """Everything the narrator may use for one turn, as plain data."""
    method_names = {s.get("method") for s in narrator.turn_log if s.get("method")} | {
        "magneto_ft_llm"
    }
    guide = {m: METHOD_GUIDE.get(m, "") for m in method_names if m in METHOD_GUIDE}
    if any(
        s.get("tool") in ("match_values", "rank_value_matches")
        and s.get("method") == "llm"
        for s in narrator.turn_log
    ):
        guide["llm (value matching)"] = VALUE_LLM_GUIDE

    decisions = []
    for src in s_changed:
        m = state.schema.get(src) or {}
        tgt = m.get("target")
        who = state.schema_by.get(src, "")
        cands = narrator.candidates.get(src, {})
        others = sorted(
            ((t, c) for t, c in cands.items() if t != tgt),
            key=lambda x: -_f(x[1]["score"]),
        )[:4]
        decision = {
            "kind": "schema match",
            "source_column": src,
            "chosen_target": tgt or "(no match)",
            "score": m.get("similarity"),
            "decided_by": who,
            "detail": "full" if ("AI" in who or "user" in who) else "medium",
            "ai_own_pick": bool(tgt) and "AI" in who and tgt not in cands,
            "other_candidates": [
                {"target": t, "score": c["score"], "method": c["method"]}
                for t, c in others
            ],
        }
        if tgt:
            decision["evidence_for_chosen"] = evidence.for_pair(
                narrator.source_path, narrator.target_path, src, tgt
            )
            if others and decision["detail"] == "full":
                decision["evidence_for_runner_up"] = evidence.for_pair(
                    narrator.source_path, narrator.target_path, src, others[0][0]
                )
        decisions.append(decision)

    by_pair = {}
    for k in v_changed:
        by_pair.setdefault((k[0], k[1]), []).append(k)
    for (src, tgt), keys in by_pair.items():
        rows = [state.values[k] for k in keys if k in state.values]
        if not rows:
            continue
        who = {state.values_by.get(k, "") for k in keys}
        unmapped = sum(1 for r in rows if _missing(r.get("target_value")))
        decisions.append(
            {
                "kind": "value mapping",
                "source_column": src,
                "target_column": tgt,
                "values_mapped": len(rows),
                "not_mapped": unmapped,
                "examples": [
                    f"{r.get('source_value')} -> {r.get('target_value')} ({fmt_sim(r.get('similarity'))})"
                    for r in rows[:4]
                ],
                "decided_by": ", ".join(sorted(w for w in who if w)),
                "detail": (
                    "full" if any("AI" in w or "user" in w for w in who) else "medium"
                ),
            }
        )
    kept = [
        line
        for s in narrator.turn_log
        for line in s.get("result", [])
        if "Kept as-is" in line
    ]
    stopped = (final or "").startswith("(run failed")
    return {
        "user_request": request,
        "turn_status": (
            "stopped early before finishing: " + final[:300] if stopped else "completed"
        ),
        "steps": mark_executed(narrator.turn_log),
        "method_guide": guide,
        "decisions": decisions,
        "kept_as_is_columns": kept,
        "final_answer_to_user": (final or "")[:1500],
    }


def build_narrator_llm(model):
    if OPENAI_REASONING.match(model):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model.split("/", 1)[-1] if model.startswith("openai/") else model,
            use_responses_api=True,
            reasoning={"effort": "low"},
            max_retries=1,
            timeout=300,
        )
    from langchain_litellm import ChatLiteLLM

    return ChatLiteLLM(model=model, max_retries=1, timeout=300)


NARRATE_BATCH = (
    40  # decisions per narrator call (big datasets are split, run in parallel)
)


async def write_narrative(llm, facts):
    """Returns (markdown, input tokens, output tokens, seconds)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    decisions = facts["decisions"]
    batches = [
        decisions[i : i + NARRATE_BATCH]
        for i in range(0, len(decisions), NARRATE_BATCH)
    ] or [[]]

    async def one(i, batch):
        part = dict(facts, decisions=batch)
        if i > 0:  # later batches: only the per-column explanations
            part = {k: part[k] for k in ("method_guide", "decisions")}
            part["note"] = "Only write the 'Why each match' items for these decisions."
        msg = await llm.ainvoke(
            [
                SystemMessage(NARRATOR_PROMPT),
                HumanMessage(json.dumps(part, default=str)),
            ]
        )
        inp, out, _ = extract_usage(msg)
        return _flatten_content(msg.content) or "", inp, out

    t0 = time.time()
    parts = await asyncio.gather(*(one(i, b) for i, b in enumerate(batches)))
    text = "\n\n".join(p[0].strip() for p in parts if p[0].strip())
    return text, sum(p[1] for p in parts), sum(p[2] for p in parts), time.time() - t0


# ---------------------------------------------------------------------------
# Mapping state: what is currently decided, who decided it, what changed
# (the MCP server holds the mappings; this mirrors them into state_<id>.json)
# ---------------------------------------------------------------------------
COMMAND_HELP = """Quick commands (no LLM, 0 tokens) - use quotes for names with spaces:
  /state                              show the current mappings
  /set <column> <target>              force a schema match (redoes that column's values if it had any)
  /rerun <column> <method>            re-match one column with another method, e.g. /rerun Grade magneto_ft_bp
  /value <column> "<value>" "<target value>"   force one value mapping
  /rerunvalues <column> <method>      redo the value matching of one column, e.g. /rerunvalues Race tfidf
  /harmonize <output folder> [file name]       write the harmonized CSV
  /source <dataset path> [target]     set the dataset for commands (normally picked up automatically)
  /help                               show this list
Anything else is sent to the agent, e.g. "map Tumor_Site to primary_site" or "now do value matching"."""


def _key(v):
    return "" if v is None else str(v)


class MappingState:
    def __init__(self, path, thread_id, agent_model):
        self.path = path
        self.thread_id = thread_id
        self.agent_model = agent_model
        self.schema = {}  # source -> {"target", "similarity"}
        self.values = {}  # (source, target, str(source_value)) -> row
        self.schema_by = {}  # source -> who decided
        self.values_by = {}  # (source, target, str(source_value)) -> who decided
        self.revisions = []

    @staticmethod
    def from_snapshot(snap):
        schema = {
            r["source_attribute"]: {
                "target": r.get("target_attribute"),
                "similarity": r.get("similarity"),
            }
            for r in snap.get("schema_matches", [])
        }
        values = {
            (
                r["source_attribute"],
                r["target_attribute"],
                _key(r.get("source_value")),
            ): r
            for r in snap.get("value_matches", [])
        }
        return schema, values

    def diff(self, new_schema, new_values):
        """Plain-English list of changes + the keys that changed."""
        lines, s_changed, v_changed = [], [], []
        for src in sorted(set(self.schema) | set(new_schema)):
            old = (self.schema.get(src) or {}).get("target")
            new = (new_schema.get(src) or {}).get("target")
            if _key(old) != _key(new):
                s_changed.append(src)
                sim = fmt_sim((new_schema.get(src) or {}).get("similarity"))
                lines.append(
                    f"- Schema: **{src}**: {old or '(no match)'} -> {new or '(no match)'} ({sim})"
                )
        for k in sorted(set(self.values) | set(new_values)):
            old = (self.values.get(k) or {}).get("target_value")
            new = (new_values.get(k) or {}).get("target_value")
            if _key(old) != _key(new):
                v_changed.append(k)
        # group value changes per column so the trace stays readable
        by_col = {}
        for k in v_changed:
            by_col.setdefault((k[0], k[1]), []).append(k)
        for (src, tgt), keys in by_col.items():
            added = all(k not in self.values for k in keys)
            removed = all(k not in new_values for k in keys)
            more = f" ... +{len(keys) - 8} more" if len(keys) > 8 else ""
            if removed:
                lines.append(
                    f"- Values [{src} -> {tgt}]: {len(keys)} old mapping(s) removed (target changed)"
                )
                continue
            sample = [
                (
                    f"'{k[2]}' -> {_key((new_values.get(k) or {}).get('target_value'))}"
                    if added
                    else f"'{k[2]}': {_key((self.values.get(k) or {}).get('target_value')) or '-'} -> "
                    f"{_key((new_values.get(k) or {}).get('target_value')) or '-'}"
                )
                for k in keys[:8]
            ]
            label = "added" if added else "changed"
            lines.append(
                f"- Values [{src} -> {tgt}] ({len(keys)} {label}): "
                + "; ".join(sample)
                + more
            )
        return lines, s_changed, v_changed

    def apply(
        self,
        new_schema,
        new_values,
        s_changed,
        v_changed,
        who_schema,
        who_value,
        request,
        by,
        turn,
    ):
        for src in s_changed:
            self.schema_by[src] = who_schema(src)
        for k in v_changed:
            self.values_by[k] = who_value(k)
        self.schema, self.values = new_schema, new_values
        # forget provenance of things that no longer exist
        self.values_by = {k: v for k, v in self.values_by.items() if k in self.values}
        if s_changed or v_changed:
            self.revisions.append(
                {
                    "turn": turn,
                    "by": by,
                    "request": request,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "schema_changed": s_changed,
                    "values_changed": [list(k) for k in v_changed],
                }
            )

    def save(self, source_path, target_path):
        columns = []
        for src, m in self.schema.items():
            vals = [
                {
                    "source_value": r.get("source_value"),
                    "target_value": r.get("target_value"),
                    "similarity": r.get("similarity"),
                    "decided_by": self.values_by.get(k, ""),
                }
                for k, r in self.values.items()
                if k[0] == src and k[1] == m.get("target")
            ]
            columns.append(
                {
                    "source": src,
                    "target": m.get("target"),
                    "similarity": m.get("similarity"),
                    "decided_by": self.schema_by.get(src, ""),
                    "value_mappings": vals,
                }
            )
        data = {
            "thread_id": self.thread_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "agent_llm": self.agent_model,
            "source_dataset": source_path,
            "target": target_path,
            "columns": columns,
            "revisions": self.revisions,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def table(self):
        if not self.schema:
            return "No mappings yet."
        rows = [
            "| Source | Target | Score | Decided by | Value maps |",
            "|---|---|---|---|---|",
        ]
        for src, m in self.schema.items():
            n = sum(1 for k in self.values if k[0] == src and k[1] == m.get("target"))
            rows.append(
                f"| {src} | {m.get('target') or '(no match)'} | {fmt_sim(m.get('similarity'))} | "
                f"{self.schema_by.get(src, '')} | {n or ''} |"
            )
        return "\n".join(rows)


def who_from_tools(turn_tools, by_user_label=None):
    """Build 'who decided' functions from the tool calls of one turn."""

    def who_schema(src):
        if by_user_label:
            return by_user_label
        for name, a in reversed(turn_tools):
            if name == "update_schema_matching" and a.get("source_attribute") == src:
                return "AI-corrected"
        for name, a in reversed(turn_tools):
            if name == "match_schema":
                return f"tool: match_schema ({a.get('method') or TOOL_DEFAULT_METHOD['match_schema']})"
        return "agent"

    def who_value(k):
        if by_user_label:
            return by_user_label
        for name, a in reversed(turn_tools):
            if (
                name == "update_value_matching"
                and a.get("source_attribute") == k[0]
                and _key(a.get("source_value")) == k[2]
            ):
                return "AI-corrected"
        for name, a in reversed(turn_tools):
            if name == "match_values":
                return f"tool: match_values ({a.get('method') or TOOL_DEFAULT_METHOD['match_values']})"
        return "agent"

    return who_schema, who_value


class Commands:
    """Quick commands: call the MCP tools directly, no LLM. Every call is traced."""

    def __init__(self, tools, narrator, state, callbacks):
        self.tools = {t.name: t for t in tools}
        self.narrator = narrator
        self.state = state
        self.callbacks = callbacks

    async def call(self, name, args, traced=True):
        config = {"callbacks": self.callbacks} if traced else {}
        out = await self.tools[name].ainvoke(args, config=config)
        result = parse_tool_output(out)
        if (
            isinstance(result, list)
            and result
            and isinstance(result[0], str)
            and "Error executing tool" in result[0]
        ):
            raise RuntimeError(result[0])
        if isinstance(result, str) and "Error executing tool" in result:
            raise RuntimeError(result)
        return result

    def _need_source(self):
        if not self.narrator.source_path:
            raise RuntimeError(
                "No dataset yet. Ask the agent to match a dataset first, or use /source <path>."
            )
        return self.narrator.source_path, self.narrator.target_path or "gdc"

    def _target_of(self, col):
        m = self.state.schema.get(col)
        if not m or _missing(m.get("target")):
            raise RuntimeError(
                f"'{col}' has no schema match. Columns: {', '.join(self.state.schema) or 'none'}"
            )
        return m["target"]

    async def _redo_values_if_needed(self, col, new_target, had_values, notes):
        if had_values and self.narrator.value_method:
            src, tgt = self._need_source()
            if target_allowed_values(tgt, new_target) == []:
                notes.append(
                    f"{new_target} has no fixed list of values, so {col}'s values are kept as-is"
                )
                return
            await self.call(
                "match_values",
                {
                    "source_dataset_path": src,
                    "target_dataset_path": tgt,
                    "attribute_matches": [
                        {"source_attribute": col, "target_attribute": new_target}
                    ],
                    "method": self.narrator.value_method,
                },
            )
            notes.append(
                f"re-ran value matching for {col} -> {new_target} with {self.narrator.value_method}"
            )

    async def run(self, argv):
        """Returns (message for the user, note for the agent or None)."""
        cmd, rest = argv[0].lower(), argv[1:]
        notes = []
        if cmd == "/set" and len(rest) == 2:
            col, target = rest
            had_values = any(k[0] == col for k in self.state.values)
            await self.call(
                "update_schema_matching",
                {
                    "source_attribute": col,
                    "new_target_attribute": target,
                    "new_similarity": 1.0,
                },
            )
            await self._redo_values_if_needed(col, target, had_values, notes)
            return (
                f"Set {col} -> {target}." + (f" Also {notes[0]}." if notes else ""),
                f"{col} mapped to {target} (forced)",
            )
        if cmd == "/rerun" and len(rest) == 2:
            col, method = rest
            src, tgt = self._need_source()
            ranked = await self.call(
                "rank_schema_matches",
                {
                    "source_dataset_path": src,
                    "target_dataset_path": tgt,
                    "attributes": [col],
                    "top_k": 3,
                    "method": method,
                },
            )
            ranked = [
                r
                for r in (ranked if isinstance(ranked, list) else [])
                if isinstance(r, dict)
            ]
            if not ranked:
                raise RuntimeError(f"{method} returned no candidates for {col}.")
            best = max(
                ranked,
                key=lambda r: (
                    _f(r.get("similarity")) if r.get("similarity") is not None else 0
                ),
            )
            had_values = any(k[0] == col for k in self.state.values)
            old = (self.state.schema.get(col) or {}).get("target")
            await self.call(
                "update_schema_matching",
                {
                    "source_attribute": col,
                    "new_target_attribute": best["target_attribute"],
                    "new_similarity": float(best.get("similarity") or 0),
                },
            )
            if _key(old) != _key(best["target_attribute"]):
                await self._redo_values_if_needed(
                    col, best["target_attribute"], had_values, notes
                )
            msg = f"{col}: {method} picked {best['target_attribute']} ({fmt_sim(best.get('similarity'))}), was {old}."
            return (
                msg + (f" Also {notes[0]}." if notes else ""),
                f"{col} re-matched with {method} -> {best['target_attribute']}",
            )
        if cmd == "/value" and len(rest) == 3:
            col, sval, tval = rest
            target = self._target_of(col)
            # use the stored source value (keeps numbers as numbers)
            stored = next(
                (
                    r.get("source_value")
                    for k, r in self.state.values.items()
                    if k[0] == col and k[2] == sval
                ),
                sval,
            )
            await self.call(
                "update_value_matching",
                {
                    "source_attribute": col,
                    "target_attribute": target,
                    "source_value": stored,
                    "new_target_value": tval,
                    "new_similarity": 1.0,
                },
            )
            return (
                f"[{col}] '{sval}' -> '{tval}'.",
                f"value '{sval}' of {col} set to '{tval}' (forced)",
            )
        if cmd == "/rerunvalues" and len(rest) == 2:
            col, method = rest
            src, tgt = self._need_source()
            target = self._target_of(col)
            if target_allowed_values(tgt, target) == []:
                return (
                    f"{target} has no fixed list of values (e.g. numbers), so {col} is not value-matched; "
                    "its values are kept as-is."
                ), None
            await self.call(
                "match_values",
                {
                    "source_dataset_path": src,
                    "target_dataset_path": tgt,
                    "attribute_matches": [
                        {"source_attribute": col, "target_attribute": target}
                    ],
                    "method": method,
                },
            )
            self.narrator.value_method = method
            return (
                f"Re-ran value matching for {col} -> {target} with {method}.",
                f"values of {col} redone with {method}",
            )
        if cmd == "/harmonize" and len(rest) in (1, 2):
            src, _ = self._need_source()
            folder = os.path.abspath(os.path.expanduser(rest[0]))
            os.makedirs(folder, exist_ok=True)
            name = (
                rest[1]
                if len(rest) == 2
                else f"harmonized_{os.path.splitext(os.path.basename(src))[0]}.csv"
            )
            res = await self.call(
                "materialize_mapping",
                {
                    "source_dataset_path": src,
                    "output_folder_path": folder,
                    "file_name": name,
                },
            )
            path = os.path.join(folder, name)
            return (
                f"Harmonized data saved to {path}",
                f"harmonized data written to {path}",
            )
        if cmd == "/source" and len(rest) in (1, 2):
            self.narrator.source_path = os.path.abspath(os.path.expanduser(rest[0]))
            if len(rest) == 2:
                self.narrator.target_path = rest[1]
            return (
                f"Dataset for commands: {self.narrator.source_path} -> {self.narrator.target_path}",
                None,
            )
        raise RuntimeError(
            f"Unknown or incomplete command: {' '.join(argv)}\n\n{COMMAND_HELP}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def run(args):
    system_prompt, prompt_src = load_system_prompt()
    if not args.plain:
        system_prompt += TRACE_INSTRUCTIONS
    effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    thread_id = args.thread_id or str(uuid.uuid4())
    trace_dir = args.trace_dir or TRACES_DIR
    os.makedirs(trace_dir, exist_ok=True)
    # files are named <dataset>_<YYYY-MM-DD>_<HH-MMAM>; until the dataset is known: session_...
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y-%m-%d_%I-%M%p")
    base = _free_base(trace_dir, f"session_{stamp}")
    jsonl_path = os.path.join(trace_dir, f"{base}.jsonl")
    md_path = os.path.join(trace_dir, f"{base}.md")
    state_path = os.path.join(trace_dir, f"{base}_state.json")

    mcp_model = args.mcp_llm or args.llm
    raw_logger = JSONLTraceLogger(jsonl_path, thread_id=thread_id)
    narrator = NarrativeTracer(md_path, args.llm, mcp_model)
    narrator._w(
        f"# BDI-Kit agent trace",
        "",
        f"- Thread: `{thread_id}`",
        f"- Started: {started.strftime('%Y-%m-%d %I:%M %p %Z')}",
        f"- Agent LLM: `{args.llm}` (reasoning effort: {args.reasoning_effort})",
        f"- LLM used inside matchers (MCP server): `{mcp_model}`",
        f"- Narrator LLM (writes the plain-English explanation): "
        + ("off" if args.no_narrative else f"`{args.narrator_llm}`"),
        f"- System prompt loaded from: `{prompt_src}`"
        + (
            " (plain, no extra instructions)"
            if args.plain
            else " + step explanations + method guide"
        ),
        f"- Files: this trace (.md), raw event log (.jsonl), current mappings (_state.json)",
    )

    llm = build_llm(args.llm, effort)
    narrator_llm = None if args.no_narrative else build_narrator_llm(args.narrator_llm)
    evidence = Evidence()
    if args.mcp_command:
        cmd, cmd_args = args.mcp_command, ["--llm", mcp_model]
    else:  # works whenever bdikit is importable; no need for bdikit-mcp on PATH
        cmd = sys.executable
        cmd_args = [
            "-c",
            "from bdikit.mcp_server import run_mcp_server; run_mcp_server()",
            "--llm",
            mcp_model,
        ]
    server = StdioServerParameters(command=cmd, args=cmd_args, env=dict(os.environ))
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [raw_logger, narrator],
        "recursion_limit": args.max_steps,
    }

    print(
        f"[agent] agent={args.llm}  matchers={mcp_model}  "
        f"narrator={'off' if args.no_narrative else args.narrator_llm}"
    )
    print(f"[agent] traces in {trace_dir} (named after the dataset once it is known)")

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"[agent] loaded {len(tools)} MCP tools")

            # agent's tools: required `why` argument (not in --plain mode) + big results
            # summarized for the model; quick commands use the original tools
            def current_pairs():
                return [
                    {
                        "source_attribute": r["source_attribute"],
                        "target_attribute": r["target_attribute"],
                    }
                    for r in narrator.mirror_schema
                    if not _missing(r.get("target_attribute"))
                ]

            agent_tools = [
                agent_tool(t, add_why=not args.plain, current_pairs=current_pairs)
                for t in tools
            ]
            agent = create_react_agent(
                llm,
                agent_tools,
                prompt=system_prompt,
                checkpointer=InMemorySaver(),
                pre_model_hook=sanitize_messages_for_llm,
            )

            state = MappingState(state_path, thread_id, args.llm)
            commands = Commands(tools, narrator, state, config["callbacks"])
            pending_notes = []  # manual changes the agent hasn't been told about yet

            async def record_changes(request, by, user_label=None):
                """Compare the mappings after this turn with the last snapshot, log + save."""
                new_schema, new_values = MappingState.from_snapshot(narrator.snapshot())
                lines, s_changed, v_changed = state.diff(new_schema, new_values)
                who_s, who_v = who_from_tools(narrator.turn_tools, user_label)
                state.apply(
                    new_schema,
                    new_values,
                    s_changed,
                    v_changed,
                    who_s,
                    who_v,
                    request,
                    by,
                    narrator.turn,
                )
                state.save(narrator.source_path, narrator.target_path)
                narrator.write_changes(lines)
                return s_changed, v_changed

            named = {"done": False}

            def name_files_after_dataset(text=""):
                """Rename session_<time> files to <dataset>_<time> once the dataset is known."""
                if named["done"]:
                    return
                path = narrator.source_path
                if not path:
                    m = re.search(r"([^/\\\s'\"]+)\.csv", text or "")
                    path = m.group(0) if m else None
                if not path:
                    return
                stem = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    os.path.splitext(os.path.basename(path))[0],
                )
                new_base = _free_base(trace_dir, f"{stem}_{stamp}")
                moves = [
                    (raw_logger, "trace_path", f"{new_base}.jsonl"),
                    (narrator, "md_path", f"{new_base}.md"),
                    (state, "path", f"{new_base}_state.json"),
                ]
                for obj, attr, fname in moves:
                    old, new = getattr(obj, attr), os.path.join(trace_dir, fname)
                    if os.path.exists(old):
                        os.replace(old, new)
                    setattr(obj, attr, new)
                named["done"] = True
                print(
                    f"[agent] trace files: {os.path.join(trace_dir, new_base)}.md / .jsonl / _state.json"
                )

            async def narrate(request, s_changed, v_changed, final):
                """Plain-English explanation of this turn, written from the logged facts."""
                if narrator_llm is None or not any(
                    s.get("kind") == "tool call" for s in narrator.turn_log
                ):
                    return  # nothing was done (e.g. just chatting): nothing to explain
                facts = build_turn_facts(
                    narrator, state, evidence, request, s_changed, v_changed, final
                )
                try:
                    text, inp, out, secs = await write_narrative(narrator_llm, facts)
                except Exception as e:
                    narrator._w(
                        "### Explanation",
                        "",
                        f"_Explanation not available: {short(str(e), 300)}_",
                        "",
                    )
                    return
                narrator._w(
                    f"### Explanation for non-experts (written by `{args.narrator_llm}` "
                    f"from the steps above; {inp} in / {out} out tokens, {secs:.1f}s)",
                    "",
                    text or "_The narrator returned nothing._",
                    "",
                )
                raw_logger._write(
                    "narrative",
                    None,
                    None,
                    model=args.narrator_llm,
                    text=text,
                    input_tokens=inp,
                    output_tokens=out,
                    facts=facts,
                )

            def show(final):
                if args.no_trace_print:
                    print("\n" + final + "\n")
                else:  # show the trace right in the conversation (it ends with the final answer)
                    print("\n" + "=" * 30 + " TRACE " + "=" * 30)
                    print(narrator.turn_text())
                    print("=" * 67 + "\n")

            async def run_turn(text):
                if text.strip().lower() == "/help":
                    print(COMMAND_HELP)
                    return
                if text.strip().lower() == "/state":
                    print(state.table())
                    print(f"(saved in {state.path})")
                    return
                if text.startswith("/"):
                    try:
                        argv = shlex.split(text)
                    except ValueError as e:
                        print(f"Could not read the command: {e}")
                        return
                    narrator.begin_turn(text, manual=True)
                    try:
                        final, note = await commands.run(argv)
                        if note:
                            pending_notes.append(note)
                    except Exception as e:
                        final = f"Command failed: {e}"
                    s_ch, v_ch = await record_changes(
                        text, "user", user_label=f"user ({argv[0]})"
                    )
                    name_files_after_dataset()
                    await narrate(text, s_ch, v_ch, final)
                    narrator.end_turn(final)
                    show(final)
                    return

                name_files_after_dataset(text)
                narrator.begin_turn(text)
                message = text
                if pending_notes:
                    message = (
                        "[Changes the user already made] "
                        + "; ".join(pending_notes)
                        + "\n\n"
                        + text
                    )
                    pending_notes.clear()
                final = ""
                try:
                    result = await agent.ainvoke({"messages": message}, config)
                    for m in reversed(result.get("messages", [])):
                        if isinstance(m, AIMessage):
                            final = _flatten_content(m.content) or ""
                            break
                except Exception as e:
                    final = f"(run failed: {e})"
                    print(f"[agent] error: {e}")
                s_ch, v_ch = await record_changes(text, "agent")
                name_files_after_dataset()
                await narrate(text, s_ch, v_ch, final)
                narrator.end_turn(final)
                show(final)

            prompts = list(args.prompt or [])
            if args.prompts_file:
                with open(args.prompts_file) as f:
                    prompts += [ln.strip() for ln in f if ln.strip()]
            if prompts:
                for p in prompts:
                    print(f"\n>>> {p}")
                    await run_turn(p)
            else:
                print(
                    "\nBDI-Kit agent ready. Ask anything, e.g. 'Match the schema of /path/data.csv to GDC'."
                )
                print(
                    "Flow: schema matching -> value matching -> harmonization. Each answer is followed by its trace."
                )
                print("Type /help for quick commands (no LLM), 'exit' to quit.")
                while True:
                    try:
                        text = input("\nYou: ").strip()
                    except EOFError:
                        break
                    if text.lower() in ("exit", "quit"):
                        break
                    if text:
                        await run_turn(text)

    print(f"[agent] done. Plain-English trace: {narrator.md_path}")
    return narrator.md_path, raw_logger.trace_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BDI-Kit agent with plain-English traces")
    p.add_argument(
        "--llm",
        default="o4-mini",
        help="Agent LLM. OpenAI reasoning models (o4-mini, o3, gpt-5...) use the Responses API; "
        "anything else goes through LiteLLM, e.g. anthropic/claude-sonnet-4-5, "
        "gemini/gemini-2.5-pro, deepinfra/openai/gpt-oss-120b (default: o4-mini)",
    )
    p.add_argument(
        "--mcp-llm",
        default=None,
        help="LLM used inside the matchers (default: same as --llm)",
    )
    p.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["none", "low", "medium", "high"],
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Use chatbot.py's prompt only (no 'Why:' lines or method guide) - for comparison runs",
    )
    p.add_argument(
        "--prompt", action="append", help="Prompt to run (repeat for multiple turns)"
    )
    p.add_argument("--prompts-file", help="File with one prompt per line")
    p.add_argument(
        "--trace-dir",
        default=None,
        help=f"Where to write traces (default: {TRACES_DIR})",
    )
    p.add_argument("--thread-id", default=None)
    p.add_argument(
        "--mcp-command",
        default=None,
        help="Use this MCP server command instead (e.g. bdikit-mcp)",
    )
    p.add_argument(
        "--max-steps", type=int, default=150, help="Max agent graph steps per turn"
    )
    p.add_argument(
        "--no-trace-print",
        action="store_true",
        help="Only print the answer (trace is still saved to the .md/.jsonl files)",
    )
    p.add_argument(
        "--narrator-llm",
        default="gpt-5-mini",
        help="LLM that writes the plain-English explanation of each turn (default: gpt-5-mini)",
    )
    p.add_argument(
        "--no-narrative",
        action="store_true",
        help="Don't write the plain-English explanation (saves the narrator's tokens)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
