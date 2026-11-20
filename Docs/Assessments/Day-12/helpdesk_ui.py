import streamlit as st
from dotenv import load_dotenv
import asyncio
from app import (
    VectorIndex, chunk_by_section, generate_embedding, mcp_call,
    _check_connection, _list_tools, mcp_tools_to_claude_schema,
    add_user_message, add_assistant_message, text_from_message, run_tools,
)
from anthropic import Anthropic
import voyageai

model = "claude-haiku-5-5"
file_for_rag = "./it_sector.txt"

async def run_conversation(client, messages, tools):
    while True:
        response = client.messages.create(
            model=model, max_tokens=1000, messages=messages, tools=tools
        )
        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            return text_from_message(response)

        add_user_message(messages, await run_tools(response))

def main():
    
    load_dotenv()
    st.title("IT HelpDesk Assistant")
    
    with open(file_for_rag, "r") as f:
        text = f.read()

    chunks = chunk_by_section(text)

    anthropic_client = Anthropic()

    print("Checking MCP connection...")
    print(asyncio.run(mcp_call(_check_connection)))

    print("Available MCP tools:")
    mcp_tools_result = asyncio.run(mcp_call(_list_tools))

    for tool in mcp_tools_result.tools:
        print(f"- {tool.name}: {tool.description}")
        print(f"  {tool.inputSchema}")
        
    claude_tools = mcp_tools_to_claude_schema(mcp_tools_result.tools)

    if "api_messages" not in st.session_state:
        st.session_state.api_messages = []
    messages = st.session_state.api_messages

    # DB C [client module] /S [remote]
    voyage_client = voyageai.Client()

    embeddings = generate_embedding(chunks, voyage_client, input_type="document")

    store = VectorIndex()
    store.add_documents([{"content": chunk} for chunk in chunks], vectors=embeddings)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input("Type your query here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        query_embedding = generate_embedding(query, voyage_client, input_type="query")
    
        results = store.search(query_embedding, k=2)
        context = "\n\n---\n\n".join(doc["content"] for doc, _ in results)
    
        prompt = f"""Use the following context to answer the question. If the context doesn't contain the answer, say so.
    
        <context>
        {context}
        </context>
    
        Question: {query}"""
        
        add_user_message(messages, prompt)

        reply = asyncio.run(run_conversation(anthropic_client, messages, tools=claude_tools))
        # keep history compact: store the plain question instead of the RAG-wrapped prompt
        for m in messages:
            if m["role"] == "user" and m["content"] == prompt:
                m["content"] = query
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == '__main__':
    main()
