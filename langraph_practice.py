# High-level pipeline (GraphRAG / Hybrid RAG)
# Load raw documents → split into chunks.
# Use a specialized prompt (LLM) to extract entities and relationships (triples).
# Ingest triples into Neo4j (nodes & relationships). Optionally add the original chunk as a node property.
# Create embeddings for chunks (OpenAI/HF) and store them either as:
# Vector index (FAISS) and link chunk → nodes, or
# Neo4j Vector Index (if using Neo4j 5+ vector features). 
# Graph Database & Analytics
# Use hybrid retrieval:
# If question is relation/graphy → use GraphCypherQAChain to generate Cypher and return structured answers.
# If question needs textual context → use vector retrieval (FAISS or Neo4j vectors) to fetch chunks and feed to an LLM.
# Optionally combine results (chain-of-thought / synthesize) before returning final answer to user.


# Raw Text
#    ↓
# LLM → Extract Triples → Neo4j
#    ↓
# Chunk + Embed → Weaviate
#    ↓
# Query → GraphCypherQAChain (Neo4j) + Semantic Search (Weaviate)
#    ↓
# LLM Synthesizes Final Answer

import os, json
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import GraphCypherQAChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.graphs import Neo4jGraph
from langchain_weaviate import Weaviate
import weaviate
from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://4799003d.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "PGdrQZkl_LcdAk25s4nRiH1Wpx4wS07Duhp6xKUGn5s"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL='q5dpy53tul9uomy3mnq.c0.asia-southeast1.gcp.weaviate.cloud'
WEAVIATE_CREDS = 'VUxvRGVyUVVsKzJYK2NaQV9GZGxVL0E2Nlg2YTRnY2ZkTTZua3JlOG1HM0RCVXZsSStSdUlueDVMN0lvPV92MjAw'


string_str = """ Jamie Dimon said he would put the chance of a serious 
fall in the US market at ‘more like 30%’, 
when 10% is currently priced in. 
Photograph: Mike Segar/ReutersView image in fullscreen
Jamie Dimon said he would put the chance of a serious fall in the US market at ‘more like 30%’,
when 10% is currently priced in. 
Photograph: Mike Segar/ReutersJP MorganHead of largest US bank warns of risk of 
American stock market crashJamie Dimon, chair of JPMorgan Chase, 
said he was ‘far more worried than others’ 
about serious market correction
Simon GoodleyThu 9 Oct 2025 12.30 BSTLast modified on Fri 10 Oct 2025 08.20 
BSTShareThe chance of the US stock market crashing is far greater than many financiers believe, 
the head of America’s largest bank has said.
Jamie Dimon, who is the chair and chief executive of the giant Wall Street bank JPMorgan Chase,
said he was “far more worried than others” about a serious market correction, 
which he predicted could come in the next six months to two years.
“I would give it a higher probability than I think is probably priced in the market and by others,”
he told the BBC.
“So if the market’s pricing in 10%,
I would say it is more like 30%.”
Dimon added there were a “lot of things out there” 
creating an atmosphere of uncertainty, 
pointing to risks including the geopolitical environment, 
fiscal spending and the remilitarisation of the world.
“All these things cause a lot of issues that we don’t know how to answer,” he said. 
“So I say the level of uncertainty should be higher in most people’s minds than what 
I would call normal.”The comments are the latest in a string of warnings that stock markets 
may be due a correction.On Wednesday, the head of the International Monetary Fund, 
Kristalina Georgieva, said the world economy had shown surprising resilience in the face of 
Donald Trump’s trade war, but issued a stark warning about the mounting risks, saying: 
“Buckle up: uncertainty is the new normal.”“Before anyone heaves a big sigh of relief, 
please hear this: global resilience has not yet been fully tested. 
And there are worrying signs the test may come,” 
she told an audience at the Milken Institute in Washington.Meanwhile, 
concerns are increasingly being aired that a stock market bubble has been created by 
high valuations of AI companies, with the Bank of England stating on Wednesday 
that there is a growing risk of a “sudden correction” in global markets.
skip past newsletter promotionSign up to Business TodayFree daily newsletterGet 
set for the working day – we'll point you to all the business news and analysis 
you need every morningEnter your email address Sign upPrivacy Notice: Newsletters 
may contain information about charities, online ads, and content funded by outside parties. 
If you do not have an account, we will create a guest account for you on 
theguardian.com to send you this newsletter. You can complete full registration at any time. 
For more information about how we use your data see our Privacy Policy. 
We use Google reCaptcha to protect our website and the Google Privacy Policy 
and Terms of Service apply.after newsletter promotionDimon conceded that some of the money 
being invested in AI would “probably be lost”.He added: “The way I look at it is AI is real;
 AI in total will pay off – just like cars in total paid off, and TVs in total paid off, 
 but most people involved in them didn’t do well.”Explore more on these topics
 JP MorganInternational Monetary Fund (IMF)International tradeBankingGlobal economyJamie 
 DimonnewsShareReuse this content"""

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
graph = Neo4jGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASS)

# === INIT ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

