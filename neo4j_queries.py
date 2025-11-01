# so we have docs right - we need to load these doc into neo4j
# to build the knowledge graph.

# To begin with - we could init our llm and embeddings which with
# extract the subject relation and object the graph database
# that's how entities(nodes) and relation and subject matters are
# arrenged in knowledge graph.
# next step is using the text eextract the triplets that we defined earlier


# now while ingesting document which need to load the content from the text
# recursively split the text
# start a neo4j driver session and extract the triples
# now the triples would be in list of dic format so we have merge them with setting up values 

# that was about injecting the document now its time to query the data from knowledge
# when we query data from a knowledge graph we mainly explore the nodes and relation of the node
# with entitities
# The next step is to explore Graph Reasoning Algorithms in Neo4j
# with these we can reason about the following stuffs
# Influence → Who are the most central or connected entities?
# Communities → Which entities form tightly linked groups?
# Similarity → How closely are topics or people related?
# Paths → How concepts propagate through the network?  


import asyncio
import json
import os
import pandas as pd
from typing import Dict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import AsyncGraphDatabase



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
neo4j_driver = None
llm = None
embeddings = None
def init_neo4j():
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    return neo4j_driver


def init_llm_and_embeddings():
    global llm, embeddings
    if llm is None:
        llm = ChatOpenAI(temperature=0)  # uses OPENAI_API_KEY from env
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    return llm, embeddings


# next step is to define how to extract the enitities out of the content

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

    # Build prompt from string (not from another PromptTemplate)
    prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT_TEMPLATE)

    # Build chain correctly with RunnableSequence
    chain = RunnableSequence(prompt | llm)

    # Invoke correctly: pass dict, not raw string
    result = chain.invoke({"text": text})

    # For newer LangChain, result is usually a ChatMessage object
    raw = getattr(result, "content", str(result))
    print("Raw LLM output:", raw)

    # --- Try parsing JSON ---
    try:
        triples = json.loads(raw)
        if isinstance(triples, list):
            return triples
    except Exception:
        pass



async def ingest_document(path: str, source: str = "local"):
    # 1. load
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    # 2. chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    driver = init_neo4j()
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

        await tx.commit()

# that was about injecting the document now its time to query the data from knowledge
# when we query data from a knowledge graph we mainly explore the nodes and relation of the node
# with entitities
# The next step is to explore Graph Reasoning Algorithms in Neo4j
# with these we can reason about the following stuffs
# Influence → Who are the most central or connected entities?
# Communities → Which entities form tightly linked groups?
# Similarity → How closely are topics or people related?
# Paths → How concepts propagate through the network?  
          
async def main():
    driver = init_neo4j()
    try:
        await ingest_document("./article.txt")
    finally:
        if driver:
            await driver.close()

from neo4j import GraphDatabase

# --- Neo4j connection setup ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")  # update with actual credentials

driver = GraphDatabase.driver(URI, auth=AUTH)
# --- Step 1: Create in-memory graph projection for reasoning ---
def create_projection(session):
    session.run("""
    CALL gds.graph.project(
      'llmSemanticGraph',
      'Entity',
      '*'
    )
    """)

# --- Step 2: Run PageRank ---
def run_pagerank(session):
    result = session.run("""
    CALL gds.pageRank.stream('llmSemanticGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS Entity, score
    ORDER BY score DESC LIMIT 10
    """)
    return pd.DataFrame(result.data())

# --- Step 3: Run Louvain Community Detection ---
def run_louvain(session):
    result = session.run("""
    CALL gds.louvain.stream('llmSemanticGraph')
    YIELD nodeId, communityId
    RETURN gds.util.asNode(nodeId).name AS Entity, communityId
    ORDER BY communityId ASC
    """)
    return pd.DataFrame(result.data())

# --- Step 4: Run Node Similarity ---
def run_similarity(session):
    result = session.run("""
    CALL gds.nodeSimilarity.stream('llmSemanticGraph')
    YIELD node1, node2, similarity
    RETURN 
        gds.util.asNode(node1).name AS Entity1,
        gds.util.asNode(node2).name AS Entity2,
        similarity
    ORDER BY similarity DESC LIMIT 10
    """)
    return pd.DataFrame(result.data())

# --- Step 5: Find Example Shortest Paths ---
def run_shortest_path(session, start_entity="Jamie Dimon", end_entity="Bank of England"):
    result = session.run(f"""
    MATCH path = shortestPath(
      (a:Entity {{name: '{start_entity}'}})-[*..5]-(b:Entity {{name: '{end_entity}'}})
    )
    RETURN [n IN nodes(path) | n.name] AS ReasoningChain
    """)
    return pd.DataFrame(result.data())

# --- Step 6: Run all reasoning algorithms and return combined insights ---
def run_all_reasoning():
    with driver.session() as session:
        # create_projection(session)

        pagerank_df = run_pagerank(session)
        louvain_df = run_louvain(session)
        similarity_df = run_similarity(session)
        paths_df = run_shortest_path(session)

    print("\n=== Influence (PageRank) ===")
    print(pagerank_df)
    print("\n=== Communities (Louvain) ===")
    print(louvain_df.head(10))
    print("\n=== Similarities (Node Similarity) ===")
    print(similarity_df)
    print("\n=== Reasoning Path Example ===")
    print(paths_df)

    # Combine for export
    return {
        "pagerank": pagerank_df,
        "louvain": louvain_df,
        "similarity": similarity_df,
        "paths": paths_df
    }

if __name__ == "__main__":
    # asyncio.run(main())
    insights = run_all_reasoning()