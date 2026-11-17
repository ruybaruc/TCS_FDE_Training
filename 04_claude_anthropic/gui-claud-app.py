import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel

load_dotenv()
client = Anthropic()

st.title("My Claude App")

# Sidebar with model selection
MODELS = [
    "claude-fable-5-2",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-haiku-4-5-20251001",
]

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Claude model", MODELS, index=MODELS.index("claude-haiku-4-5-20251001"))
    thinking_enabled = st.toggle("Enable thinking", value=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your prompt here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate bot response
    response = client.messages.create(
        model=selected_model,
        max_tokens=4096,
        thinking=(
            {"type": "enabled", "budget_tokens": 1024}
            if thinking_enabled
            else {"type": "disabled"}
        ),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # With thinking on, the response also contains thinking blocks; keep only the text
    reply = "".join(block.text for block in response.content if block.type == "text")
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})