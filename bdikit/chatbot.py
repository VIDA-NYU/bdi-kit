import os
import sys

# Relaunch with streamlit if not already running inside streamlit
if os.environ.get("STREAMLIT_RUN") != "1":
    os.environ["STREAMLIT_RUN"] = "1"

    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            __file__,
            "--",
            *sys.argv[1:],
        ]
    )
    sys.exit(0)
    import os
import re
import queue
import asyncio
import argparse
import threading
import time
import litellm
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver

litellm.suppress_debug_info = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--llm", default="deepinfra/openai/gpt-oss-120b", help="LLM model name"
)
args, _ = parser.parse_known_args()

llm_model = args.llm

# Page configuration
st.set_page_config(page_title="BDI-Kit AI Agent", page_icon="🤖", layout="centered")

st.title("🤖 BDI-Kit AI Agent")
st.markdown("Integrate and harmonize your datasets")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_loop" not in st.session_state:
    st.session_state.agent_loop = None
    st.session_state.agent_thread = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "initializing" not in st.session_state:
    st.session_state.initializing = False

if "init_start_time" not in st.session_state:
    st.session_state.init_start_time = None

if "request_queue" not in st.session_state:
    st.session_state.request_queue = queue.Queue()
    st.session_state.response_queue = queue.Queue()

if "processed_message_count" not in st.session_state:
    st.session_state.processed_message_count = 0

if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

server_params = StdioServerParameters(
    command="bdikit-mcp", args=["--llm", llm_model], env=dict(os.environ)
)

prompt = """
You are a tool-augmented assistant specialized in dataset integration and harmonization.
You can perform tasks such as schema matching, value matching, and materializing outputs.
You have access to a set of tools. Use them to perform actions, preferably one at a time.

You support two main workflows:
- Schema Matching
- Value Matching


========================
SCHEMA MATCHING WORKFLOW
========================

For schema matching tasks, you may follow this typical workflow:

1. Run `match_schema` to generate initial attribute matches.

2. Review the matches and identify potential issues. You should flag matches when:
   - Similarity is low (e.g., 0.5 or lower)
   - Attribute names appear semantically unrelated
   - There is ambiguity between multiple plausible matches
   - The match appears inconsistent with domain knowledge

3. When a match is flagged, you SHOULD call at least one of the following tools:
   - `preview_domain` to inspect example values
   - `rank_schema_matches` to retrieve alternative matches (consider top_k=10)

   Choose the most appropriate tool:
   - Use `preview_domain` when semantic meaning depends on values
   - Use `rank_schema_matches` when searching for better attribute candidates
   - You may call both if needed, but avoid unnecessary calls

4. After gathering additional information:
   - Decide whether to keep the original match or correct it
   - Automatic corrections are allowed when clearly better
   - Keep track of the original match for provenance

5. Present the final results in a single markdown table:
See the found matches:

| Source Attribute | Target Attribute | Status | Reason |
|------------------|------------------|--------|--------|
| <source> | <target> | ✅ OK / ⚠️ AI-corrected | <very short reason> |

Guidelines:
- Skip similarity values in this table
- Keep reasons very short
- If AI-corrected, mention the original target attribute in the reason
- If the AI-corrected target attribute is the same as the original match, mark it as ✅ OK 
- Only correct when necessary
- Avoid overwhelming the user

Examples:

See the found matches:
| Source Attribute | Target Attribute | Status | Reason |
|------------------|------------------|--------|--------|
| participant_country | tumor_grade | ⚠️ AI-corrected | corrected from *diagnosis_method*; low similarity |
| gender | sex | ✅ OK | high semantic match |
| bmi | bmi | ✅ OK | exact match |


========================
VALUE MATCHING WORKFLOW
========================

For value matching tasks, you may follow this typical workflow:

1. Run `match_values` to generate initial value matches.

2. Review the matches and identify potential issues. You should flag matches when:
   - Similarity is low (e.g., 0.5 or lower)
   - The matches appear semantically incorrect
   - The matches contradicts domain knowledge

3. When a match is flagged, you SHOULD call at least one of the following tools:
   - `preview_domain` to inspect source and target values
   - `rank_value_matches` to retrieve alternative value candidates (consider top_k=5 or 10)

Choose the most appropriate tool:
- Use `preview_domain` when semantic meaning depends on domain values
- Use `rank_value_matches` when searching for better value matches
- You may call both if needed, but avoid unnecessary calls

4. After gathering additional information:
   - Decide whether to keep the original match or correct it
   - Automatic corrections are allowed when clearly better
   - Keep track of the original match for provenance

5. Present the final results in a markdown table:

See the found matches:

| Source Value | Target Value | Status | Reason |
|--------------|--------------|--------|--------|
| <source> | <target> | ✅ OK / ⚠️ AI-corrected | <very short reason> |

Guidelines:
- Skip similarity values in this table
- Keep reasons very short
- If AI-corrected, mention the original target value
- Only correct when necessary
- Avoid overwhelming the user

Examples:

See the found matches:

| Source Value | Target Value | Status | Reason |
|--------------|--------------|--------|--------|
| pancreatic carcinoma | Cancer Related | ✅ OK | semantic match |
|  heart disorder | Cardiovascular Disorder | ⚠️ AI-corrected | corrected from *Infection* |
| other: prostate carcinoma | Cancer Related | ✅ OK | cancer subtype |


========================
PROVENANCE (WHY?)
========================

If the user asks "why?" about a correction:
Provide a concise provenance explanation using a simple vertical arrow flow with markdown.
Avoid long paragraphs.

Use the following format:
The following provenance graph explains the correction:

participant_country
  ↓
call match_schema() → diagnosis_method (0.10) ❌
  ↓
call preview_domain() → USA, Canada, Germany
  ↓
call rank_schema_matches()
  ↓
country (0.82) ✅ selected

After the flow, include a short explanation sentence:

Example:

"Selected 'country' because domain values indicate general country names and it had the best semantic match among alternatives."

Guidelines:
- Always wrap the flow inside a markdown code block (```)
- Use vertical arrows (↓) for readability
- Use explicit tool calls (e.g., call match_schema())
- Include similarity scores when available


========================
USER CONTROL
========================

Allow the user to accept, modify, or override any proposed match.


========================
WHAT-IF CONSTRAINTS
========================

Users may specify additional constraints for schema or value matching.
These constraints represent "what-if scenarios" that modify the matching behavior.

When a user provides constraints, you MUST:

1. Interpret the constraint clearly
2. Update the schema or value matches accordingly
3. Identify which matches changed due to the constraint
4. Explain the consequences concisely

Constraints may include (but are not limited to):

Schema Matching Constraints:
- Each source attribute maps to at most one target attribute
- Certain attributes must (or must not) match
- Enforce domain-specific mapping rules

Value Matching Constraints:
- Certain source values must map only to specific target values
- Prefer exact or standardized mappings
- Domain-specific mapping rules

After applying constraints, present the updated results:

See the updated matches after applying constraints:

| Source | Target | Status | Reason |
|--------|--------|--------|--------|
| <source> | <target> | ✅ OK / 🔄 Updated | <short reason> |

Guidelines:
- Use 🔄 Updated only when the constraint changed the match
- Keep explanations short
- Preserve unchanged matches as ✅ OK
- Avoid overwhelming the user

Example:

User constraint:
"Each source value must map to a unique target value"

See the updated matches after applying constraints:

| Source Value | Target Value | Status | Reason |
|--------------|--------------|--------|--------|
| male | Male | ✅ OK | <original_reason> |
| na | Not Reported | 🔄 Updated | constraint enforced |
| not available | Unknown | 🔄 Updated | constraint enforced |

After the table, include a short explanation sentence:

Example:

"Mapping updated to enforce one-to-one correspondence between source and target values."


========================
IMPORTANT BEHAVIOR
========================

Important:
- If you recommend running a tool, you should call it instead of only suggesting it
- Prefer calling one tool at a time
- Do not repeat tool calls indefinitely


========================
RESPONSE FORMAT
========================

Response format:
- If calling a tool, output a JSON object with `tool` and `parameters`
- Otherwise respond with markdown tables and explanations
- Suggest next steps when applicable
"""


async def agent_worker(request_queue, response_queue):
    """Background worker that maintains the async context"""
    llm_kwargs = {}

    # If the model name has a @ in this format, it's from Portkey, send the appropriate configuration
    if re.search(r".+/@.+", llm_model):
        llm_kwargs = {
            "api_base": os.getenv("PORTKEY_API_BASE"),
            "extra_headers": {"x-portkey-api-key": os.getenv("PORTKEY_API_KEY")},
        }
    try:
        print(f"[AGENT] Initializing with model: {llm_model}")
        llm = ChatLiteLLM(
            model=llm_model, streaming=True, max_retries=1, timeout=300, **llm_kwargs
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                print(f"[AGENT] Loaded {len(tools)} tools")
                agent = create_react_agent(
                    llm, tools, prompt=prompt, checkpointer=checkpointer
                )

                # Signal initialization complete
                response_queue.put(("initialized", None))
                print("[AGENT] Initialization complete")

                # Process requests
                while True:
                    try:
                        try:
                            message = request_queue.get_nowait()
                            if message == "STOP":
                                break

                            print(f"[AGENT] Processing message: {message}")

                            # Process the message
                            result = await agent.ainvoke({"messages": message}, config)

                            messages = result.get("messages", [])
                            print(f"[AGENT] Got {len(messages)} messages in response")

                            for i, msg in enumerate(messages):
                                print(f"[AGENT] Message {i}: {type(msg).__name__}")
                                if hasattr(msg, "content"):
                                    print(
                                        f"[AGENT]   Content: {str(msg.content)[:200]}"
                                    )

                            response_queue.put(("success", messages))
                            print("[AGENT] Response queued")

                        except queue.Empty:
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"[AGENT] Error processing message: {e}")
                        import traceback

                        traceback.print_exc()
                        response_queue.put(("error", str(e)))
    except Exception as e:
        print(f"[AGENT] Fatal error: {e}")
        import traceback

        traceback.print_exc()
        response_queue.put(("init_error", str(e)))


def run_agent_loop(request_queue, response_queue):
    """Run the agent in a separate thread with its own event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(agent_worker(request_queue, response_queue))
    finally:
        loop.close()


def format_message_for_display(msg):
    """Format different message types for display"""
    formatted_messages = []

    if isinstance(msg, AIMessage):
        # Handle tool calls
        if msg.additional_kwargs.get("tool_calls"):
            tool_info = []
            for call in msg.additional_kwargs["tool_calls"]:
                tool_info.append(f"**Calling tool**: {call['function']['name']}")
                tool_info.append(f"**Args**: `{call['function']['arguments']}`")
            # formatted_messages.append(("tool", "\n".join(tool_info)))
            print("\n".join(tool_info))

        # Handle regular content
        if msg.content.strip():
            formatted_messages.append(("assistant", msg.content))

    elif isinstance(msg, ToolMessage):
        # formatted_messages.append(("tool_response", f"**Tool Response**:\n{msg.content}"))
        print(f"**Tool Response**:\n{msg.content}")

    return formatted_messages


# Initialize agent on first run
if not st.session_state.initialized and not st.session_state.initializing:
    # Start initialization
    st.session_state.initializing = True
    st.session_state.init_start_time = time.time()

    # Start the agent thread
    agent_thread = threading.Thread(
        target=run_agent_loop,
        args=(st.session_state.request_queue, st.session_state.response_queue),
        daemon=True,
    )
    agent_thread.start()
    st.session_state.agent_thread = agent_thread
    print("[UI] Agent thread started, rerunning to check status")
    st.rerun()

# Check initialization status (non-blocking)
if st.session_state.initializing:
    with st.spinner("Initializing assistant..."):
        try:
            msg_type, msg_data = st.session_state.response_queue.get_nowait()

            if msg_type == "initialized":
                st.session_state.initialized = True
                st.session_state.initializing = False
                st.success("Agent initialized!")
                print("[UI] Agent initialized successfully")
                time.sleep(0.5)
                st.rerun()

            elif msg_type == "init_error":
                st.error(f"Initialization error: {msg_data}")
                st.session_state.initializing = False
                st.stop()

        except queue.Empty:
            # Check for timeout
            if time.time() - st.session_state.init_start_time > 30:
                st.error("Initialization timeout. Please refresh the page.")
                st.session_state.initializing = False
                st.stop()
            else:
                # Still waiting, rerun to check again
                print("[UI] Still waiting for initialization...")
                time.sleep(0.1)
                st.rerun()

    st.stop()  # Don't show the rest of the UI while initializing

# Only show the UI after initialization is complete
if st.session_state.initialized:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.write(content)
            elif role == "tool":
                with st.chat_message("assistant"):
                    st.info(content)
            elif role == "tool_response":
                with st.chat_message("assistant"):
                    st.success(content)

        # Show thinking indicator INSIDE the chat container
        if st.session_state.waiting_for_response:
            with st.chat_message("assistant"):
                st.markdown("🤔 Thinking...")

    # Input area - ALWAYS AT THE BOTTOM
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question:",
            placeholder="Ask to integrate datasets, match schemas, etc.",
            key="user_input",
        )
        submit_button = st.form_submit_button("Send 📤")

    # Process user input - QUEUE IT AND SHOW IMMEDIATELY
    if submit_button and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Send request to agent
        st.session_state.request_queue.put(user_input)
        print(f"[UI] Queued message: {user_input}")

        # Set flag to check for response later
        st.session_state.waiting_for_response = True

        # IMPORTANT: Rerun immediately to display the user message
        st.rerun()

# OUTSIDE the form submission, check for responses
if st.session_state.waiting_for_response:
    print("[UI] Waiting for response, checking queue")

    # Now check for response
    try:
        # Use a small timeout so the spinner shows
        msg_type, msg_data = st.session_state.response_queue.get(timeout=1.0)
        print(f"[UI] Received from queue: {msg_type}")

        if msg_type == "success":
            all_messages = msg_data
            new_messages = all_messages[st.session_state.processed_message_count :]

            print(f"[UI] Processing {len(new_messages)} new messages")

            for msg in new_messages:
                # Skip HumanMessage
                if isinstance(msg, HumanMessage):
                    print(f"[UI] Skipping HumanMessage")
                    continue

                formatted = format_message_for_display(msg)
                print(f"[UI] Formatted into {len(formatted)} display messages")

                for msg_type, msg_content in formatted:
                    print(f"[UI] Adding to history: {msg_type}")
                    if msg_type == "tool":
                        st.session_state.messages.append(
                            {"role": "tool", "content": msg_content}
                        )
                    elif msg_type == "tool_response":
                        st.session_state.messages.append(
                            {"role": "tool_response", "content": msg_content}
                        )
                    elif msg_type == "assistant":
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg_content}
                        )

            st.session_state.processed_message_count = len(all_messages)
            st.session_state.waiting_for_response = False
            print("[UI] Response processed, rerunning to display")
            st.rerun()

        elif msg_type == "error":
            st.error(f"Error: {msg_data}")
            st.session_state.waiting_for_response = False

    except queue.Empty:
        # Still waiting, rerun to keep checking
        print("[UI] Still waiting for response, rerunning...")
        time.sleep(0.3)  # Small delay so user sees the thinking message
        st.rerun()


def run_streamlit():
    import sys
    from streamlit.web import cli as stcli

    script_path = __file__

    # Set up sys.argv as if streamlit was called from command line
    sys.argv = ["streamlit", "run", script_path, "--"] + sys.argv[1:]

    # Run streamlit directly in the same process (no subprocess)
    sys.exit(stcli.main())
