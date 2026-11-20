import math
from datetime import datetime
from typing import Optional, Any, List, Dict, Tuple
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession


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
    
def chunk_by_section(document_text):
    import re

    pattern = r"\n## "
    return re.split(pattern, document_text)

def generate_embedding(chunks, voyage_client, model="voyage-3-large", input_type="document"):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]
    result = voyage_client.embed(input, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]

async def mcp_call(fn):
    """Open a fresh MCP connection, run fn(session), then close the connection."""
    async with streamable_http_client("http://localhost:8000/mcp") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await fn(session)

async def _check_connection(session):
    return "connected"

async def _list_tools(session):
    return await session.list_tools()

def mcp_tools_to_claude_schema(mcp_tools):
    return [
        {
            "name": tool.name,
            "description": tool.description or tool.name,
            "input_schema": tool.input_schema,
        }
        for tool in mcp_tools
    ]

def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, message):
    messages.append({"role": "assistant", "content": message.content})


def text_from_message(message):
    return "\n".join(block.text for block in message.content if block.type == "text")


async def run_tool(tool_name, tool_input):
    # Local (non-MCP) tools
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)

    async def _call(session):
        return await session.call_tool(tool_name, tool_input)

    result = await mcp_call(_call)
    return "\n".join(block.text for block in result.content if block.type == "text")


async def run_tools(message):
    tool_result_blocks = []
    for block in message.content:
        if block.type != "tool_use":
            continue
        try:
            output = await run_tool(block.name, block.input)
            tool_result_blocks.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": False}
            )
        except Exception as e:
            tool_result_blocks.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {e}", "is_error": True}
            )
    return tool_result_blocks

def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)
