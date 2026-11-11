import streamlit as st
from calc_components import (
    CALCULATOR_PROMPT,
    CALCULATOR_TOOLS,
    ask,
    build_calculator_agent,
    get_model,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from pydantic import BaseModel

class CalcAnswer(BaseModel):
    result: float
    explanation: str

agent = build_calculator_agent(checkpointer=InMemorySaver())
#agent = build_calculator_agent(checkpointer=InMemorySaver(), response_format=CalcAnswer)

st.title("Calculator Chat Bot")

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
   
    response = ask(agent, prompt, thread_id="demo", verbose=False)['messages'][-1].content
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})