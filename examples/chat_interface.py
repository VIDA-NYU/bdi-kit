import os
import sys
import asyncio
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver
import threading
import queue


# Check sys.argv for parameters
for i, arg in enumerate(sys.argv):
    if arg == "--llm" and i + 1 < len(sys.argv):
        llm_model = sys.argv[i + 1]


# Default values if not provided
if not llm_model:
    llm_model = "deepinfra/openai/gpt-oss-120b1"


# Page configuration
st.set_page_config(
    page_title="AI Agent to use BDI-Kit", page_icon="🤖", layout="centered"
)

st.title("🤖 AI Agent to use BDI-Kit")
st.markdown("Integrate and harmonize your datasets")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_loop" not in st.session_state:
    st.session_state.agent_loop = None
    st.session_state.agent_thread = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "request_queue" not in st.session_state:
    st.session_state.request_queue = queue.Queue()
    st.session_state.response_queue = queue.Queue()

if "processed_message_count" not in st.session_state:
    st.session_state.processed_message_count = 0

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

server_params = StdioServerParameters(
    command="bdikit-mcp", args=["--llm", llm_model], env=dict(os.environ)
)

prompt = """
You are a tool-augmented assistant specialized in dataset integration and harmonization.
In dataset integration, you may need to perform tasks such as schema matching, value matching and the materialization of the outputs.
You have access to a set of tools. Use the tools to perform actions, preferable one at a time.
Do not include tutorials how to solve the task, just the explanations of the outputs from the tools.
Respond in the following format:
- If calling a tool, output a JSON object with the tool name and parameters.
- If providing a final answer to the user, output it as plain text after tool calls.
If applicable, suggest next steps to the users.

Important:
- Do not repeat the same tool calls indefinitely.
- After a tool call, interpret its output and decide if further tool use is actually necessary. Try to make only one call per user request.
"""


async def agent_worker(request_queue, response_queue):
    """Background worker that maintains the async context"""
    try:
        llm = ChatLiteLLM(model=llm_model, temperature=0.7)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(
                    llm, tools, prompt=prompt, checkpointer=checkpointer
                )

                # Signal initialization complete
                response_queue.put(("initialized", None))

                # Process requests
                while True:
                    try:
                        # Check for new requests (non-blocking)
                        try:
                            message = request_queue.get_nowait()
                            if message == "STOP":
                                break

                            # Process the message
                            result = await agent.ainvoke({"messages": message}, config)
                            messages = result.get("messages", [])
                            response_queue.put(("success", messages))
                        except queue.Empty:
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        response_queue.put(("error", str(e)))
    except Exception as e:
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
if not st.session_state.initialized:
    with st.spinner("Initializing assistant..."):
        # Start the agent thread
        agent_thread = threading.Thread(
            target=run_agent_loop,
            args=(st.session_state.request_queue, st.session_state.response_queue),
            daemon=True,
        )
        agent_thread.start()
        st.session_state.agent_thread = agent_thread

        # Wait for initialization
        import time

        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                msg_type, msg_data = st.session_state.response_queue.get_nowait()
                if msg_type == "initialized":
                    st.session_state.initialized = True
                    st.success("Agent initialized!")
                    break
                elif msg_type == "init_error":
                    st.error(f"Initialization error: {msg_data}")
                    st.stop()
            except queue.Empty:
                time.sleep(0.1)

        if not st.session_state.initialized:
            st.error("Initialization timeout. Please refresh the page.")
            st.stop()

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

# Input area
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question:",
        placeholder="Ask to integrate datasets, match schemas, etc.",
        key="user_input",
    )
    submit_button = st.form_submit_button("Send 📤")

# Process user input
if submit_button and user_input.strip():
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Send request to agent
    st.session_state.request_queue.put(user_input)

    # Wait for response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            import time

            timeout = 120
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    msg_type, msg_data = st.session_state.response_queue.get(timeout=1)

                    if msg_type == "success":
                        # Process and display agent messages
                        # Only process new messages (not already displayed)
                        all_messages = msg_data
                        new_messages = all_messages[
                            st.session_state.processed_message_count :
                        ]

                        for msg in new_messages:
                            formatted = format_message_for_display(msg)
                            for msg_type, msg_content in formatted:
                                if msg_type == "tool":
                                    st.info(msg_content)
                                    st.session_state.messages.append(
                                        {"role": "tool", "content": msg_content}
                                    )
                                elif msg_type == "tool_response":
                                    st.success(msg_content)
                                    st.session_state.messages.append(
                                        {
                                            "role": "tool_response",
                                            "content": msg_content,
                                        }
                                    )
                                elif msg_type == "assistant":
                                    st.write(msg_content)
                                    st.session_state.messages.append(
                                        {"role": "assistant", "content": msg_content}
                                    )

                        # Update the count of processed messages
                        st.session_state.processed_message_count = len(all_messages)
                        break
                    elif msg_type == "error":
                        st.error(f"Error: {msg_data}")
                        break
                except queue.Empty:
                    continue

    # Rerun to update the display
    st.rerun()

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
    This agent helps with:
    - **Schema matching**: Aligning columns between datasets
    - **Value matching**: Mapping values across datasets
    - **Data materialization**: Creating integrated outputs

    Simply ask your question and the agent will use the appropriate tools to help you.
    """
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.processed_message_count = 0
        st.rerun()

    st.markdown("---")
    st.caption("Powered by BDI-Kit")
