import streamlit as st
from dotenv import load_dotenv
from app import VectorIndex, chunk_by_section, generate_embedding 
from anthropic import Anthropic
import voyageai

def main():
    
    load_dotenv()
    st.title("IT HelpDesk Assistant")
    file_for_rag = "./it_sector.txt"

    with open(file_for_rag, "r") as f:
        text = f.read()

    chunks = chunk_by_section(text)

    model = "claude-haiku-5-5"
    anthropic_client = Anthropic()

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
    
        # Generate bot response
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
        #next(block.text for block in response.content if block.type == "text")
    
        # With thinking on, the response also contains thinking blocks; keep only the text
        reply = "".join(block.text for block in response.content if block.type == "text")
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == '__main__':
    main()
