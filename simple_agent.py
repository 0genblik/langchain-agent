import os 
import time 
import json 
import requests
from pathlib import Path
from typing import List, Dict, Any
import faiss
import pickle 
from sentence_transformers import SentenceTransformer

# langchain yay
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool 
from langchain_tavily import TavilySearch

import gradio as gr # UI/frontend library 

# load api key with dotenv
from dotenv import load_dotenv
load_dotenv()

# define RAG constants
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
INDEX_PATH = "chunks.index"
CHUNKS_PATH = "chunks.pkl"


# 1. Gemini client with throttle + retry 
class GeminiClient:
    """Wrapper over the Gemini+Langchain model with throttle and exponential backoff"""
    def __init__(self, model: str = "gemini-2.0-flash-lite", min_interval: float = 0.5, max_retries: int = 5):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)
        self._last_call_time = 0.0
        self.min_interval = min_interval
        self.max_retries = max_retries

    def _throttle(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_time = time.time()
    
    def invoke_with_retry(self, messages: List[Any]) -> AIMessage:
        """Invoke the Gemini model with a retry on errors only"""
        backoff = 1.0 
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                return self.llm.invoke(messages)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e
        

# 2. Minimal chain function (prompt -> response)
def minimal_chain(client: GeminiClient, question: str) -> str:
    msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question)
    ]
    response = client.invoke_with_retry(msgs)
    return response.content


# 3. RAG integration 
# LLMs have limitations - training data might be outdated, or they might not know niche topics.
# Precise *factual* detail -> often hallucinates.
# Models don't know our private information.
# We could put this information into the prompt - it doesn't scale. 

# RAG := retrieval-augmented generation
# 1. Query arrives
# 2. System retrieves from a vector index the top-k relevant document chunks
# 3. Retrieved text is "context injected" into the prompt to the model.
# 4. Model generated an answer *grounded* in the retrieved text. 

# => less likely to hallucinate
# => answer questions about private data

def load_documents(folder: str) -> List[Dict[str, Any]]:
    docs = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"doc_id": str(p), "text": text, "meta": {"source": str(p)}})
        elif p.suffix.lower() == ".pdf":
            # For simplicity, we skip PDF parsing in this example
            continue
        elif p.suffix.lower() == ".csv":
            # For simplicity, we skip CSV parsing in this example
            continue
        elif p.suffix.lower() == ".json":
            # For simplicity, we skip JSON parsing in this example
            continue
    return docs
    
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def build_or_load_index(docs_folder: str):
    if Path(INDEX_PATH).exists() and Path(CHUNKS_PATH).exists():
        # Load existing index and chunks
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    docs = load_documents(docs_folder)
    all_chunks = []
    for doc in docs:
        cks = chunk_text(doc["text"])
        for i, ck in enumerate(cks):
            all_chunks.append({"chunk_id": f"{doc['doc_id']}_chunk_{i}", "text": ck, "meta": doc["meta"]})
    
    embeddings = EMBEDDER.encode([c["text"] for c in all_chunks], convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    return index, all_chunks

index, all_chunks = build_or_load_index("data")

# 4. Tools 

# @tool
# def search_web(query: str) -> str:
#     """
#     Search the web for a query using DuckDuckGo Instant Answer API and return a summary.
    
#     Args:
#         query (str): The search query.
#     Returns:
#         str: A summary of the search results.
#     """
#     print("=====================================================")
#     print(f"Searching the web for: {query}")
#     url = "https://api.duckduckgo.com/"
#     params = {
#         "q": query,
#         "format": "json",
#         "t": "langchain_agent",
#     }
#     r = requests.get(url, params=params, timeout=10)
#     print(r)
#     r.raise_for_status()
#     data = r.json()
#     print(f"Search results: {data}")
#     print("=====================================================")
#     summary = data.get("AbstractText") or ""
#     if not summary:
#         rel = data.get("RelatedTopics") or []
#         if rel and isinstance(rel, list):
#             first = rel[0]
#             if isinstance(first, dict):
#                 summary = first.get("Text") or ""
#                 summary = summary[:500] + " (See more at: " + first.get("FirstURL", "") + ")"
#     return summary or "No relevant information found."


tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

@tool
def wiki_lookup(topic: str) -> str:
    """
    Get the first paragraph from Wikipedia for a given topic.
    
    Args:
        topic (str): The Wikipedia topic to look up. You should use underscores for spaces.
    Returns:
        str: The first paragraph of the Wikipedia article or an error message.
    """
    print("=====================================================")
    print(f"Looking up Wikipedia for: {topic}")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    # Add User-Agent header to prevent 403 errors
    headers = {
        "User-Agent": "LangchainAgent/1.0 (https://github.com/0genblik/langchain-agent)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()  # Raises exception for 4XX/5XX responses
        data = r.json()
        print(f"Wikipedia data: {data}")
        print("=====================================================")
        return data.get("extract", "No summary available.")
    except requests.exceptions.HTTPError as e:
        error_msg = f"Wikipedia lookup failed with status code {e.response.status_code}"
        if e.response.status_code == 403:
            error_msg += ". This could be due to rate limiting or request blocking."
        elif e.response.status_code == 404:
            error_msg += f". The topic '{topic}' might not exist on Wikipedia."
        print(error_msg)
        print("=====================================================")
        return error_msg
    except Exception as e:
        error_msg = f"Wikipedia lookup failed: {str(e)}"
        print(error_msg)
        print("=====================================================")
        return error_msg
    


@tool
def retrieve_relevant_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks for a query using a FAISS vector index.
    
    Args:
        query (str): The search query.
        k (int): The number of top relevant documents to retrieve.
    Returns:
        List[Dict[str, Any]]: A list of the top-k relevant document chunks with their metadata.
    """
    print("=====================================================")
    print(f"Retrieving top {k} relevant documents for query: {query}")
    q = EMBEDDER.encode([query], convert_to_numpy=True)
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        ch = all_chunks[int(idx)]
        results.append({"chunk_id": ch["chunk_id"], "text": ch["text"], "meta": ch["meta"], "score": float(score)})

    print(f"Top {k} relevant documents: {results}")
    print("=====================================================")
    return results


# 5. Agent loop with tools - ReAct style
class AgentLoop:
    def __init__(self, client: GeminiClient, tools: List[Any], max_steps: int = 10):
        self.client = client
        self.tools = tools
        self.llm_tools = client.llm.bind_tools(tools)
        self.max_steps = max_steps

    def run(self, user_message: str) -> str:
        """Run the agent loop until a final answer is produced or max steps reached."""
        messages: List[Any] = [
            SystemMessage(content="You are a helpful agent. You should call tools when needed. Always retrieve relevant documentation first, and only if the results are poor matches should you call another tool, e.g. search. If you call a tool, output a JSON specifying tool name and arguments. Otherwise, output a final answer."),
            HumanMessage(content=user_message)
        ]

        seen = set() # to avoid repeated tool calls

        for _ in range(self.max_steps):
            ai: AIMessage = self.llm_tools.invoke(messages)
            if ai.tool_calls:
                tc = ai.tool_calls[0]  # Assume one tool call at a time
                tname, targs, tid = tc["name"], tc["args"], tc["id"]
                key = (tname, json.dumps(targs, sort_keys=True))
                if key in seen:
                    messages.append(ai)
                    messages.append(ToolMessage(tool_call_id=tid, content=json.dumps({"output": "Repeated tool call avoided."})))
                    continue 
                seen.add(key)

                tool_map = {t.name: t for t in self.tools}
                try:
                    obs = tool_map[tname].invoke(targs)
                except Exception as e:
                    obs = f"Error invoking tool {tname}: {str(e)}"
                
                messages.append(ai)
                messages.append(ToolMessage(tool_call_id=tid, content=json.dumps({"output": obs})))
                continue
            
            # no tool calls, final answer:
            print("=====================================================")
            print(f"Message history: {messages}")
            print("=====================================================")
            return str(ai.content)
        
        return "I couldn't complete the task within the allowed steps."


# Run 
if __name__ == "__main__":
    client = GeminiClient()

    question = "Tell me about principles of agentic behaviour."

    print("Minimal chain response:")
    print(minimal_chain(client, question))

    agent = AgentLoop(client, tools=[wiki_lookup, retrieve_relevant_docs])
    print("Agent loop response:")
    print(agent.run(question))