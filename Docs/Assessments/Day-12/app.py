import streamlit as st
from anthropic import Anthropic
import voyageai
import math
from typing import Optional, Any, List, Dict, Tuple

class VectorIndex:
    def __init__(self, distance_metric: str = "cosine", embedding_fn=None):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    def add_document(self, document: Dict[str, Any]):
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        vector = self._embedding_fn(document["content"])
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]], vectors: Optional[List[List[float]]] = None):
        # Bulk version of add_document, to avoid one API call per chunk.
        if vectors is None:
            if not self._embedding_fn:
                raise ValueError("Provide either precomputed vectors or an embedding function.")
            vectors = self._embedding_fn([doc["content"] for doc in documents])

        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def add_vector(self, vector, document: Dict[str, Any]):
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}")

        self.vectors.append(list(vector))
        self.documents.append(document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        # Accepts raw text (embedded on the fly) or a ready-made vector.
        # Returns (document, distance) pairs — lower distance is a better match.
        if not self.vectors:
            return []

        if isinstance(query, str):
            query_vector = self._embedding_fn(query)
        else:
            query_vector = query

        dist_func = self._cosine_distance if self._distance_metric == "cosine" else self._euclidean_distance

        distances = [(dist_func(query_vector, v), doc) for v, doc in zip(self.vectors, self.documents)]
        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    def _euclidean_distance(self, vec1, vec2) -> float:
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _cosine_distance(self, vec1, vec2) -> float:
        # 1 - cosine similarity: 0 (identical direction) to 2 (opposite).
        # Compares direction only, so a long chunk and a short chunk about the
        # same topic still score close.
        mag1 = math.sqrt(sum(x * x for x in vec1))
        mag2 = math.sqrt(sum(x * x for x in vec2))
        if mag1 == 0 or mag2 == 0:
            return 1.0
        dot = sum(p * q for p, q in zip(vec1, vec2))
        similarity = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return 1.0 - similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}')"
    
client = Anthropic()

model = "claude-haiku-5-5"
#model="claude-sonnet-5-5"

# DB C [client module] /S [remote]
voyage_client = voyageai.Client()

def chunk_by_section(document_text):
    import re

    pattern = r"\n## "
    return re.split(pattern, document_text)

with open("./it_sector.txt", "r") as f:
    text = f.read()

chunks = chunk_by_section(text)

def generate_embedding(chunks, model="voyage-3-large", input_type="document"):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]
    result = voyage_client.embed(input, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]

embeddings = generate_embedding(chunks, input_type="document")

store = VectorIndex()
store.add_documents([{"content": chunk} for chunk in chunks], vectors=embeddings)

#Modify to get query from user input
query = "What did the Phase IIa trial of the new compound show?"
query_embedding = generate_embedding(query, input_type="query")

results = store.search(query_embedding, k=2)
context = "\n\n---\n\n".join(doc["content"] for doc, _ in results)

prompt = f"""Use the following context to answer the question. If the context doesn't contain the answer, say so.

<context>
{context}
</context>

Question: {query}"""

response = anthropic_client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
)

next(block.text for block in response.content if block.type == "text")


st.title("IT HelpDesk Assistant")


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
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # With thinking on, the response also contains thinking blocks; keep only the text
    reply = "".join(block.text for block in response.content if block.type == "text")
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})