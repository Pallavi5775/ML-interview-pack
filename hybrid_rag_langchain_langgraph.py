"""
Hybrid RAG Mini-Project
LangChain + (optional) LangGraph + Neo4j + Weaviate

File: hybrid_rag_langchain_langgraph.py
Purpose: runnable scaffold demonstrating an end-to-end Hybrid RAG pipeline:
 - ingestion (chunk -> extract triples -> ingest Neo4j -> embed -> upsert Weaviate)
 - query API (FastAPI) with async parallel retrieval (Neo4j + Weaviate)
 - LLM semaphore pool, circuit breaker, Redis caching
 - Prometheus metrics instrumentation

Notes:
 - This is a scaffold for learning and testing. Replace placeholder calls with your real clients and credentials.
 - Environment variables required (example):
     OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASS, WEAVIATE_URL, REDIS_URL
 - To run locally: create a virtualenv, `pip install -r requirements.txt` (see requirements list below)

Requirements (example):
  langchain
  openai
  weaviate-client
  langchain-weaviate
  neo4j
  fastapi
  uvicorn[standard]
  aioredis
  prometheus_client
  pydantic
  tiktoken

Run:
  export OPENAI_API_KEY=... NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASS=pass WEAVIATE_URL=http://localhost:8080 REDIS_URL=redis://localhost:6379/0
  uvicorn hybrid_rag_langchain_langgraph:app --host 0.0.0.0 --port 8000 --reload

This file intentionally keeps implementations minimal and defensive. Replace the `async_llm_generate`, `async_neo4j_query`, and `async_weaviate_search` placeholders with your production-ready clients.
"""

import os
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Prometheus metrics
from prometheus_client import start_http_server, Histogram, Gauge, Counter

# Redis (async)
from redis import asyncio as aioredis

# Neo4j async driver
from neo4j import AsyncGraphDatabase

# Weaviate client (sync used here for simplicity; adapt to async wrapper if desired)
import weaviate

# LangChain (LLM & embedding placeholders)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional LangGraph import -- if unavailable, the scaffold uses a tiny fallback orchestrator
try:
    from langgraph import StateGraph, END  # hypothetical modern LangGraph API
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


import logging
import time
import json
import asyncio
from fastapi import HTTPException

# ---------------------------
# Structured Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("HybridRAG")


# ---------------------------
# Config & Env
# ---------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")



# LLM concurrency planning
LLM_CONCURRENCY_SLOTS = 40
LLM_TIMEOUT_SECONDS = 15
# Weaviate class name
WEAVIATE_CLASS = "DocumentChunk"

# ---------------------------
# Prometheus metrics
# ---------------------------
start_http_server(8001)  # Prometheus scrape endpoint (port)
REQUEST_LATENCY = Histogram("request_latency_seconds", "Total hybrid request latency")
NEO4J_LATENCY = Histogram("neo4j_latency_seconds", "Neo4j query latency")
WEAVIATE_LATENCY = Histogram("weaviate_latency_seconds", "Weaviate query latency")
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM generation latency")
IN_FLIGHT = Gauge("in_flight_requests", "Current in-flight hybrid requests")
LLM_ERRORS = Counter("llm_errors_total", "LLM failure count")
CACHE_HITS = Counter("cache_hits_total", "Cache hits for query results")
CACHE_MISSES = Counter("cache_misses_total", "Cache misses for query results")

# ---------------------------
# App + Clients (lazy init)
# ---------------------------
app = FastAPI(title="Hybrid RAG (LangChain + LangGraph) Scaffold")
redis: Optional[aioredis.Redis] = None
neo4j_driver = None
weaviate_client = None
llm = None
embeddings = None

# LLM semaphore and circuit breaker
LLM_SEMAPHORE = asyncio.Semaphore(LLM_CONCURRENCY_SLOTS)

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, test_allowance=2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.test_allowance = test_allowance
        self.fail_count = 0
        self.state = "CLOSED"
        self.opened_at = None
        self.test_in_progress = 0

    def record_success(self):
        self.fail_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.failure_threshold:
            self.state = "OPEN"
            self.opened_at = time.time()

    def allow(self):
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.opened_at > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.test_in_progress = 0
                return True
            else:
                return False
        if self.state == "HALF_OPEN":
            if self.test_in_progress < self.test_allowance:
                self.test_in_progress += 1
                return True
            else:
                return False

circuit = CircuitBreaker()

# ---------------------------
# Utility: initialize clients
# ---------------------------
async def init_redis():
    global redis
    if redis is None:
        redis = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return redis

def init_neo4j():
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    return neo4j_driver

import weaviate.classes.init as wvc_init



def init_weaviate():
    """
    Initialize Weaviate client (REST-only mode, disables gRPC fully).
    """
    weaviate_client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,           # still required for schema, but won't be used
        grpc_secure=False,
        additional_config=wvc_init.AdditionalConfig(
            timeout=wvc_init.Timeout(init=10, query=30),
            # ✅ Force all operations to use REST instead of gRPC
            use_rest_for_queries=True
        ),
        skip_init_checks=True,
    )
    return weaviate_client
# def init_weaviate():
#     global weaviate_client
#     if weaviate_client is None:
#         # Normalize environment URL (remove double ports)
#         url = os.getenv("WEAVIATE_URL", "http://localhost:8080").strip()
#         url = url.replace(":8080:8080", ":8080")

#         weaviate_client = weaviate.connect_to_custom(
#             http_host="localhost",
#             http_port=8080,
#             http_secure=False,
#             grpc_host=None,
#             grpc_port=None,
#             grpc_secure=False,
#             additional_config=wvc_init.AdditionalConfig(
#                 timeout=wvc_init.Timeout(init=10)
#             ),
#             skip_init_checks=True,  # ✅ skip gRPC ping & OIDC
#         )
#     return weaviate_client

def init_llm_and_embeddings():
    global llm, embeddings
    if llm is None:
        llm = ChatOpenAI(temperature=0)  # uses OPENAI_API_KEY from env
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    return llm, embeddings

# ---------------------------
# Schema setup for Neo4j and Weaviate (run once at startup)
# ---------------------------
import re
import weaviate
import weaviate.classes.config as wvc_config

async def setup_schema():
    """
    Create Neo4j constraints and Weaviate schema safely across all Neo4j 5.x versions.
    Detects version and chooses correct fulltext syntax automatically.
    """
    driver = init_neo4j()
    async with driver.session() as session:
        # --- Detect Neo4j version ---
        version_result = await session.run(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
        )
        record = await version_result.single()
        version_str = record["version"] if record else "5.0.0"
        match = re.match(r"(\\d+)\\.(\\d+)", version_str)
        major, minor = (int(match.group(1)), int(match.group(2))) if match else (5, 0)
        print(f"🧠 Detected Neo4j version: {version_str}")

        # --- Constraints ---
        await session.run("""
        CREATE CONSTRAINT entity_name IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.name IS UNIQUE
        """)
        await session.run("""
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.id IS UNIQUE
        """)

        # --- Fulltext index ---
        if major == 5 and minor < 9:
            # ✅ Legacy syntax (Neo4j 5.0–5.8)
            print("⚙️ Using legacy fulltext syntax (ON EACH [e.name])")
            await session.run("""
            CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
            FOR (e:Entity)
            ON EACH [e.name]
            """)
        else:
            # ✅ Modern syntax (Neo4j 5.9+)
            print("⚙️ Using modern fulltext syntax (ON (e.name))")
            await session.run("""
            CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
            FOR (e:Entity)
            ON (e.name)
            """)

    # --- Weaviate setup ---
    client = init_weaviate()
    try:
        schema = client.collections.list_all()  # v4+
        class_exists = WEAVIATE_CLASS in schema
    except AttributeError:
        # Fallback for Weaviate v3.x
        schema = client.schema.get()
        class_exists = any(c["class"] == WEAVIATE_CLASS for c in schema.get("classes", []))

    if not class_exists:
        client.collections.create(
            name=WEAVIATE_CLASS,
            vectorizer_config=wvc_config.Configure.Vectorizer.none(),
            properties=[
                wvc_config.Property(name="text", data_type=wvc_config.DataType.TEXT),
                wvc_config.Property(name="source", data_type=wvc_config.DataType.TEXT),
                wvc_config.Property(name="chunk_id", data_type=wvc_config.DataType.TEXT),
            ]
        )
        print(f"✅ Created Weaviate collection: {WEAVIATE_CLASS}")
    else:
        print(f"ℹ️ Weaviate collection '{WEAVIATE_CLASS}' already exists.")



# ---------------------------
# Simple triple extractor using LLMChain (synchronous wrapper)
# Replace with more robust extractor for production
# ---------------------------
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# --- Define your string template directly ---
EXTRACT_PROMPT_TEMPLATE = """
Extract key entity–relation–entity triples from the following text.
Return a JSON array of objects like:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Text:
{text}
"""

async def extract_triples(text: str) -> List[Dict[str, str]]:
    """
    Extracts (subject, relation, object) triples from text using an LLM.
    Handles both valid and malformed JSON gracefully.
    """
    llm, _ = init_llm_and_embeddings()

    # ✅ Build prompt from string (not from another PromptTemplate)
    prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT_TEMPLATE)

    # ✅ Build chain correctly with RunnableSequence
    chain = RunnableSequence(prompt | llm)

    # ✅ Invoke correctly: pass dict, not raw string
    result = chain.invoke({"text": text})

    # For newer LangChain, result is usually a ChatMessage object
    raw = getattr(result, "content", str(result))
    print("🔍 Raw LLM output:", raw)

    # --- Try parsing JSON ---
    try:
        triples = json.loads(raw)
        if isinstance(triples, list):
            return triples
    except Exception:
        pass

    # --- Fallback: extract JSON-like substring ---
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # --- Fallback: return empty if no structured data found ---
    return []

# ---------------------------
# Ingestion: chunk, extract, write to Neo4j and Weaviate
# ---------------------------
async def ingest_document(path: str, source: str = "local"):
    # 1. load
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    # 2. chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    driver = init_neo4j()
    client = init_weaviate()
    _, emb = init_llm_and_embeddings()

    async with driver.session() as session:
        tx = await session.begin_transaction()
        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(path)}_chunk_{i}"

            # extract triples (blocking call to LLM)
            triples = await extract_triples(chunk.page_content)

            # --- Neo4j entity + relationship upsert ---
            for t in triples:
                await tx.run(
                    """
                    MERGE (a:Entity {name:$sub})
                    MERGE (b:Entity {name:$obj})
                    MERGE (a)-[r:REL]->(b)
                    SET r.type = $rel
                    """,
                    {
                        "sub": t.get("subject"),
                        "obj": t.get("object"),
                        "rel": t.get("relation"),
                    },
                )

            # --- Chunk node upsert ---
            await tx.run(
                "MERGE (c:Chunk {id:$id}) SET c.text = $text, c.source=$source",
                {"id": chunk_id, "text": chunk.page_content, "source": source},
            )

            # --- Weaviate vector upsert ---
            vector = emb.embed_documents([chunk.page_content])[0]
            collection = client.collections.get(WEAVIATE_CLASS)
            collection.data.insert(
                properties={
                    "text": chunk.page_content,
                    "source": source,
                    "chunk_id": chunk_id,
                },
                vector=vector
            )
            # Optional batching commit
            if i % 20 == 0:
                await tx.commit()
                tx = await session.begin_transaction()

        await tx.commit()

# ---------------------------
# Async placeholders: replace with real async clients
# ---------------------------
async def async_neo4j_query(question: str) -> str:
    """
    Runs a graph lookup for relevant entities or relationships related to the question.
    """
    driver = init_neo4j()
    async with driver.session() as session:
        result = await session.run("""
            MATCH (a:Entity)-[r:REL]->(b:Entity)
            WHERE a.name CONTAINS $q OR b.name CONTAINS $q OR r.type CONTAINS $q
            RETURN a.name AS subject, r.type AS relation, b.name AS object
            LIMIT 10
        """, {"q": question})

        records = [f"{r['subject']} -[{r['relation']}]-> {r['object']}" async for r in result]

    return "\n".join(records)


async def async_weaviate_search(query: str, k: int = 5):
    _, emb = init_llm_and_embeddings()
    client = init_weaviate()
    collection = client.collections.get(WEAVIATE_CLASS)

    vector = emb.embed_query(query)

    try:
        results = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_metadata=wvc_query.MetadataQuery(distance=True)
        )
    except Exception as e:
        raise RuntimeError(f"Weaviate REST query failed: {e}")
    finally:
        client.close()

    docs = [
        {
            "text": obj.properties.get("text"),
            "source": obj.properties.get("source"),
            "distance": obj.metadata.distance,
        }
        for obj in results.objects
    ]
    return docs

# Very simple async LLM wrapper using langchain chat model (blocking under the hood) -- replace with proper async LLM client
import asyncio
from langchain_openai import ChatOpenAI

async def async_llm_generate(prompt: str) -> str:
    """
    Generate an LLM response asynchronously with:
    - Circuit breaker protection
    - Semaphore concurrency control
    - Latency + error metrics
    """
    # --- Circuit breaker guard ---
    if not circuit.allow():
        raise RuntimeError("LLM circuit open")

    # --- Acquire concurrency slot ---
    async with LLM_SEMAPHORE:
        with LLM_LATENCY.time():
            try:
                # Initialize LLM only once per call (or globally cache if needed)
                llm_obj, _ = init_llm_and_embeddings()

                # If using ChatOpenAI, prefer async API:
                if hasattr(llm_obj, "ainvoke"):
                    result = await llm_obj.ainvoke(prompt)
                else:
                    # Fallback: run blocking call in thread pool
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: llm_obj.invoke(prompt))

                # Normalize result to string
                response = getattr(result, "content", str(result))

                circuit.record_success()
                return response.strip()

            except Exception as e:
                LLM_ERRORS.inc()
                circuit.record_failure()
                raise RuntimeError(f"LLM error: {e}")


# ---------------------------
# Hybrid query flow (async gather + caching + circuit + fallback)
# ---------------------------
class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    await init_redis()
    init_neo4j()
    init_weaviate()
    init_llm_and_embeddings()
    await setup_schema()

@app.get("/health")
async def health():
    return {"status": "ok"}

import weaviate.classes.query as wvc_query
@app.post("/hybrid_query")
async def hybrid_query(req: QueryRequest):
    q = req.question.strip()
    if not q:
        logger.warning("❌ Received empty question in request")
        raise HTTPException(status_code=400, detail="Question required")

    IN_FLIGHT.inc()
    start = time.time()

    request_id = f"req-{int(start * 1000)}"
    logger.info(f"🚀 [START] [{request_id}] Query: {q}")

    # --- Redis cache lookup ---
    r = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    cache_key = f"hybrid:answer:{q}"
    cached = await r.get(cache_key)
    if cached:
        CACHE_HITS.inc()
        IN_FLIGHT.dec()
        logger.info(f"⚡ [{request_id}] Cache HIT for question: {q}")
        return {"answer": json.loads(cached), "cached": True}

    CACHE_MISSES.inc()
    logger.info(f"🧠 [{request_id}] Cache MISS. Executing hybrid retrieval...")

    # --- Run Weaviate + Neo4j concurrently ---
    neo_task = asyncio.create_task(async_neo4j_query(q))
    weav_task = asyncio.create_task(async_weaviate_search(q, k=5))

    try:
        t1 = time.time()
        neo_res, weav_res = await asyncio.gather(neo_task, weav_task)
        logger.info(f"📈 [{request_id}] Retrieval done: Neo4j={len(neo_res)} lines, Weaviate={len(weav_res)} docs, time={time.time() - t1:.2f}s")
    except Exception as e:
        logger.exception(f"❌ [{request_id}] Retrieval failed: {e}")
        IN_FLIGHT.dec()
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    # --- Compose prompt ---
    context_parts = ["Graph results:\n", neo_res, "\nTextual evidence:\n"]
    for w in weav_res:
        if w.get("text"):
            context_parts.append(w["text"])
    prompt = "\n".join(context_parts) + f"\n\nQuestion: {q}\nAnswer:"

    # --- Generate final answer with LLM + circuit breaker ---
    try:
        logger.info(f"💬 [{request_id}] Sending composed prompt ({len(prompt)} chars) to LLM...")
        t2 = time.time()
        answer = await asyncio.wait_for(async_llm_generate(prompt), timeout=LLM_TIMEOUT_SECONDS)
        logger.info(f"✅ [{request_id}] LLM completed in {time.time() - t2:.2f}s")

    except asyncio.TimeoutError:
        logger.warning(f"⏰ [{request_id}] LLM timed out after {LLM_TIMEOUT_SECONDS}s")
        IN_FLIGHT.dec()
        raise HTTPException(status_code=504, detail="LLM timeout")

    except RuntimeError:
        fallback = " ".join([w.get("text", "")[:300] for w in weav_res])
        await r.setex(cache_key, 60, json.dumps({"answer": fallback, "fallback": True}))
        logger.warning(f"⚠️ [{request_id}] Circuit open. Returned fallback answer.")
        IN_FLIGHT.dec()
        return {"answer": fallback, "fallback": True}

    except Exception as e:
        logger.exception(f"💥 [{request_id}] LLM generation failed: {e}")
        IN_FLIGHT.dec()
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # --- Cache final result ---
    await r.setex(cache_key, 300, json.dumps({"answer": answer}))
    logger.info(f"🧩 [{request_id}] Answer cached successfully (TTL=300s)")

    # --- Record metrics and finalize ---
    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)
    IN_FLIGHT.dec()
    logger.info(f"🏁 [{request_id}] Completed in {latency:.2f}s | Cached=False")

    return {"answer": answer, "cached": False}


async def async_weaviate_search(query: str, k: int = 5):
    """
    Asynchronous wrapper for Weaviate vector search (v4.x API).
    Performs a near_vector semantic search.
    """
    _, emb = init_llm_and_embeddings()
    client = init_weaviate()
    collection = client.collections.get(WEAVIATE_CLASS)

    # Generate query embedding
    vector = emb.embed_query(query)

    # Perform vector search
    results = collection.query.near_vector(
        near_vector=vector,
        limit=k,
        return_metadata=wvc_query.MetadataQuery(distance=True)
    )

    # Parse results into a list of dicts
    docs = []
    for obj in results.objects:
        docs.append({
            "text": obj.properties.get("text"),
            "source": obj.properties.get("source"),
            "distance": obj.metadata.distance
        })

    client.close()  # Prevent memory leaks
    return docs


# ---------------------------
# Minimal LangGraph fallback orchestrator (if real LangGraph not installed)
# ---------------------------
class MiniGraphRuntime:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, a, b):
        self.edges.setdefault(a, []).append(b)

    async def run(self, start_node, state):
        # naive BFS/DFS executor
        funcs = {n: self.nodes[n] for n in self.nodes}
        order = [start_node]
        while order:
            n = order.pop(0)
            fn = funcs[n]
            out = fn(state)
            if asyncio.iscoroutine(out):
                out = await out
            if isinstance(out, dict):
                state.update(out)
            for child in self.edges.get(n, []):
                order.append(child)
        return state

# Example LangGraph-like pipeline
async def example_langgraph_pipeline(text: str):
    runtime = MiniGraphRuntime()

    async def extract_node(s):
        triples = await extract_triples(s["text"])
        return {"triples": triples}

    async def ingest_node(s):
        # upsert triples to neo4j
        driver = init_neo4j()
        async with driver.session() as session:
            for t in s.get("triples", []):
                await session.run("MERGE (a:Entity {name:$sub}) MERGE (b:Entity {name:$obj}) MERGE (a)-[:REL {type:$rel}]->(b)",
                                  {"sub": t.get("subject"), "obj": t.get("object"), "rel": t.get("relation")})
        return {"ingested": True}

    runtime.add_node("extract", extract_node)
    runtime.add_node("ingest", ingest_node)
    runtime.add_edge("extract", "ingest")

    state = {"text": text}
    final = await runtime.run("extract", state)
    return final

# ---------------------------
# End of scaffold
# ---------------------------

if __name__ == "__main__":
    print("This file provides a FastAPI app and scaffold. Run via uvicorn as documented in the header.")
