# Breaks text into spaCy sentences.
# Accumulates them until the chunk reaches a target token size (e.g., 800 tokens).
# Adds overlap for continuity.
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

import os
import string
import textwrap
from openai import AsyncOpenAI, OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import tiktoken
from weaviate.embedded import EmbeddedOptions
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc

text_lines = string_str.splitlines()

max_token = 200
overlap_sentences = 2
enc = tiktoken.encoding_for_model("gpt-4-turbo")

def count_tokens(text):
    return len(enc.encode(text))

def is_punctuation(char):
    return unicodedata.category(char).startswith('P')


# So, here is the story for today - I imported one of the semi-structured data which usually
# come when you scrap from website. In my case I had scraped it earlier through a cron for iiT app 
# in the news collection.

# I decided to explore and implement proper chunking on the content, But had a said this earlier that
# i despise the idea of using regex in any of my code. simply because it difficult to memorize.

# So I used a is_punctuation method to filter punctuations through the content and clean the content of 
# the punctuations but use the is_punctuation method to split the content into splits.

# then i used a token length budget strategy to form the chunks. wherein if the curr token exceeds token 
# budget then stop sectioning and start a new chunks and repeat the process.

# Below you can visualize the quality of the chunking done so far. By analysing this graph we can
# adjust the max_token and overlap windows as well.

# | Metric              | Ideal Range            | Your Observation           | Inference                |
# | ------------------- | ---------------------- | -------------------------- | ------------------------ |
# | Token Length        | 80–100% of `max_token` | Mostly ~100, one small dip | consistent chunking   |
# | Overlap             | 20–50 words            | 20–30 words                | smooth continuity     |
# | Semantic Similarity | 0.7–0.9                | ~0.72                      | healthy semantic flow |
# | Outliers            | few                    | one                        | acceptable             |


def split_line(line):
    out = []
    temp = ""
    for ch in line:
        if is_punctuation(ch):
            if temp.strip():
                out.append(temp.strip())
            temp = ""
        else:
            temp += ch
    if temp.strip():
        out.append(temp.strip())
    return out

# --- Chunking ---
chunks = []
cur, cur_token = [], 0

for line in text_lines:
    splits = split_line(line)

    for each_split in splits:
        sent_tokens = count_tokens(each_split)
        if cur_token + sent_tokens > max_token and cur:
            chunks.append(" ".join(cur).strip())
            cur = cur[-overlap_sentences:] if overlap_sentences < len(cur) else cur
            cur_token = sum(count_tokens(x) for x in cur)
        cur.append(each_split)
        cur_token += sent_tokens

if cur:
    chunks.append(" ".join(cur).strip())

print(f"Total chunks created: {len(chunks)}")

###################################################################################################
########################## Next Step - Embedd chunks into Vector ##################################

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client_opennai = OpenAI(api_key=OPENAI_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url='q5dpy53tul9uomy3mnq.c0.asia-southeast1.gcp.weaviate.cloud',
    auth_credentials=Auth.api_key('VUxvRGVyUVVsKzJYK2NaQV9GZGxVL0E2Nlg2YTRnY2ZkTTZua3JlOG1HM0RCVXZsSStSdUlueDVMN0lvPV92MjAw'),
    headers={"X-Openai-Api-Key": OPENAI_KEY}
)

print(client.is_ready())  # Should print: `True`


vectors = []
for chunk in chunks:
    response = client_opennai.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    )
    for j, data in enumerate(response.data):
        vectors.append(data.embedding)


collection_name = "TestChunk"

if client.collections.exists(collection_name):  # In case we've created this collection before
    client.collections.delete(collection_name)  # THIS WILL DELETE ALL DATA IN THE COLLECTION

chunk_obj = client.collections.create(
    name=collection_name,
    properties=[
        wvc.config.Property(
            name="chunk",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="chunk_index",
            data_type=wvc.config.DataType.INT
        ),
    ],
    generative_config=wvc.config.Configure.Generative.openai(),
    vector_config=wvc.config.Configure.Vectors.text2vec_openai(),
)    


chunks_list = list()
for i, chunk in enumerate(chunks):
    data_properties = {
        "chunk": chunk,
        "chunk_index": i
    }
    data_object = wvc.data.DataObject(properties=data_properties)
    chunks_list.append(data_object)
chunk_obj.data.insert_many(chunks_list)

response = chunk_obj.aggregate.over_all(total_count=True)
print(response.total_count)

#generative queries 

response = chunk_obj.generate.fetch_objects(
    limit=2,
    grouped_task="Write a trivia tweet based on this text. Use emojis and make it succinct and cute."
)

print(response.generative.text)


######################################################################################################
########################################Weaviate Queries##############################################

# Search by semantic meaning using embeddings.
# You provide a vector → get top similar objects.


resp = chunk_obj.aggregate.over_all(total_count=True)
print(f"✅ Total objects in collection: {resp.total_count}\n")

# --------------------------
# 6️⃣ Ask a query (RAG style)
# --------------------------
user_query = "What did Jamie Dimon warn about regarding the U.S. stock market?"
print(f"🔍 Query: {user_query}\n")

retrieved = chunk_obj.query.hybrid(
    query=user_query,
    alpha=0.7,
    limit=3,
    return_properties=["chunk", "chunk_index"]
)

print(f"✅ Retrieved {len(retrieved.objects)} relevant chunks")

# --- Generate (separate step) ---
# Use the "grouped_task" to summarize across retrieved chunks
response = chunk_obj.generate.fetch_objects(
    limit=3,
    grouped_task="Summarize Jamie Dimon's warning in one paragraph for a tweet."
)

# --- Display ---
print("\n🧠 Generated Answer:\n")
print(textwrap.fill(response.generative.text, width=100))
print("\n" + "="*100 + "\n")

print("📚 Top Retrieved Chunks (Context):\n")
for obj in retrieved.objects:
    print(f"Chunk #{obj.properties['chunk_index']}:\n{textwrap.fill(obj.properties['chunk'], width=100)}\n")
    print("-"*100)


client.close()

# Optional Extensions

# Change grouped_task to other tasks:

# grouped_task="Write a LinkedIn-style summary of the risks mentioned."
# grouped_task="Create a bullet-point summary of Dimon’s main warnings."
# grouped_task="Write a one-sentence headline for a news post."


# Adjust alpha

# alpha=1.0 → purely semantic search

# alpha=0.0 → purely keyword search

# alpha=0.5–0.8 → balanced RAG

# Add retrieval score display:

# for obj in response.objects:
#     print(obj.metadata.score)


# | Query Type      | Purpose                  | Example Method                      |
# | --------------- | ------------------------ | ----------------------------------- |
# | Fetch all       | List objects             | `.query.fetch_objects()`            |
# | Vector search   | Semantic retrieval       | `.query.near_vector()`              |
# | Keyword search  | Literal matches          | `.query.bm25()`                     |
# | Hybrid search   | Combine vector + keyword | `.query.hybrid()`                   |
# | Filtered search | Conditional filtering    | `.query.fetch_objects(filters=...)` |
# | Aggregation     | Stats or counts          | `.aggregate.over_all()`             |
# | Generative      | Summarization / Q&A      | `.generate.fetch_objects()`         |
# | Ask query       | QA-style search          | `.query.ask()`                      |
# | RAG combo       | Search + LLM             | `.query.hybrid(...).generate(...)`  |

# # https://chatgpt.com/c/69026a95-6184-8323-b142-5bdf6899de9b

# # Try A/B testing for dynamic allocation of threshholds
# # --- Visualization ---
# np.random.seed(42)
# vectors = np.random.rand(len(chunks), 128) if len(chunks) else np.empty((0, 128))

# # 1️⃣ Token distribution
# token_lengths = [count_tokens(ch) for ch in chunks] if chunks else [0]

# # 2️⃣ Overlap count
# overlaps = []
# if len(chunks) > 1:
#     for i in range(1, len(chunks)):
#         prev, cur_set = set(chunks[i-1].split()), set(chunks[i].split())
#         overlaps.append(len(prev & cur_set))
# else:
#     overlaps = [0]

# # 3️⃣ Semantic similarity
# similarities = []
# if len(vectors) > 1:
#     for i in range(1, len(vectors)):
#         sim = cosine_similarity([vectors[i-1]], [vectors[i]])[0][0]
#         similarities.append(sim)
# else:
#     similarities = [0]

# # --- Safe averages ---
# def safe_mean(values):
#     arr = np.array(values, dtype=float)
#     return float(np.nanmean(arr)) if arr.size > 0 else 0.0

# avg_tokens = safe_mean(token_lengths)
# avg_overlap = safe_mean(overlaps)
# avg_similarity = safe_mean(similarities)
# token_var = np.std(token_lengths)

# print("\n🧠 Sample Chunk Inspection:\n" + "-"*80)
# for i, ch in enumerate(chunks[:3]):
#     print(f"Chunk {i+1}:\n{ch[:300]}...\n{'-'*80}")

# print(f"\n📊 Avg token count: {avg_tokens:.2f}")
# print(f"🔁 Avg overlap words: {avg_overlap:.2f}")
# print(f"🧠 Avg semantic similarity: {avg_similarity:.2f}")

# # --- Automated Insights ---
# if avg_tokens >= 0.8 * max_token and token_var < 0.2 * avg_tokens:
#     token_msg = f"✅ Consistent chunk sizes near limit ({avg_tokens:.0f}/{max_token})"
#     token_color = "green"
# elif avg_tokens < 0.5 * max_token:
#     token_msg = f"⚠️ Average tokens low ({avg_tokens:.0f}), consider merging"
#     token_color = "orange"
# else:
#     token_msg = f"⚠️ High variation ({token_var:.1f}), refine split rules"
#     token_color = "red"

# if 20 <= avg_overlap <= 50:
#     overlap_msg = f"✅ Overlap balanced ({avg_overlap:.1f} words)"
#     overlap_color = "green"
# elif avg_overlap > 60:
#     overlap_msg = f"⚠️ Overlap too high ({avg_overlap:.1f}), may cause redundancy"
#     overlap_color = "orange"
# else:
#     overlap_msg = f"⚠️ Overlap too low ({avg_overlap:.1f}), may lose context"
#     overlap_color = "red"

# if avg_similarity >= 0.7:
#     sim_msg = f"✅ Good semantic continuity (avg sim={avg_similarity:.2f})"
#     sim_color = "green"
# elif 0.5 <= avg_similarity < 0.7:
#     sim_msg = f"⚠️ Moderate continuity (avg sim={avg_similarity:.2f})"
#     sim_color = "orange"
# else:
#     sim_msg = f"❌ Poor continuity (avg sim={avg_similarity:.2f})"
#     sim_color = "red"

# # --- Unified Plotly Dashboard ---
# fig = make_subplots(
#     rows=3, cols=1,
#     subplot_titles=(
#         "Token Length Distribution & Sequence",
#         "Overlap Word Count Between Consecutive Chunks",
#         "Semantic Continuity (Cosine Similarity Between Adjacent Chunks)"
#     ),
#     vertical_spacing=0.12
# )

# # 1️⃣ TOKEN LENGTH
# fig.add_trace(
#     go.Histogram(x=token_lengths, nbinsx=20, marker_color='skyblue', opacity=0.6, name="Token Distribution"),
#     row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(
#         x=list(range(1, len(token_lengths)+1)),
#         y=token_lengths,
#         mode='lines+markers',
#         line=dict(color='orange', width=2),
#         name='Chunk Token Length',
#         marker=dict(size=6)
#     ),
#     row=1, col=1
# )
# fig.add_hline(
#     y=max_token,
#     line_dash="dash",
#     line_color="red",
#     annotation_text=f"Target max token = {max_token}",
#     annotation_position="top right",
#     row=1, col=1
# )

# # 2️⃣ OVERLAP
# fig.add_trace(
#     go.Scatter(
#         x=list(range(1, len(overlaps)+1)),
#         y=overlaps,
#         mode='lines+markers',
#         line=dict(color='royalblue', width=2),
#         marker=dict(size=8),
#         name="Overlap Count"
#     ),
#     row=2, col=1
# )

# # 3️⃣ SEMANTIC SIMILARITY
# fig.add_trace(
#     go.Scatter(
#         x=list(range(1, len(similarities)+1)),
#         y=similarities,
#         mode='lines+markers',
#         line=dict(color='green', width=2),
#         marker=dict(size=8),
#         name="Cosine Similarity"
#     ),
#     row=3, col=1
# )

# # --- Annotated Insights ---
# fig.add_annotation(
#     text=token_msg,
#     xref="paper", yref="paper",
#     x=0.5, y=1.07,
#     showarrow=False,
#     font=dict(color=token_color, size=13)
# )

# fig.add_annotation(
#     text=overlap_msg,
#     xref="paper", yref="paper",
#     x=0.5, y=0.69,
#     showarrow=False,
#     font=dict(color=overlap_color, size=13)
# )

# fig.add_annotation(
#     text=sim_msg,
#     xref="paper", yref="paper",
#     x=0.5, y=0.31,
#     showarrow=False,
#     font=dict(color=sim_color, size=13)
# )

# # --- Layout ---
# fig.update_layout(
#     height=950,
#     showlegend=False,
#     title_text="📊 Chunking Performance & Continuity Dashboard (with AI Insights)",
#     title_x=0.5,
#     template="plotly_white",
#     margin=dict(t=80, b=40),
# )

# # Axis labels
# fig.update_xaxes(title_text="Chunk Index", showgrid=True, row=1, col=1)
# fig.update_yaxes(title_text="Token Count", showgrid=True, row=1, col=1)
# fig.update_xaxes(title_text="Chunk Index (i → i+1)", showgrid=True, row=2, col=1)
# fig.update_yaxes(title_text="Overlap Word Count", showgrid=True, row=2, col=1)
# fig.update_xaxes(title_text="Chunk Index (i → i+1)", showgrid=True, row=3, col=1)
# fig.update_yaxes(title_text="Cosine Similarity", showgrid=True, row=3, col=1, range=[0, 1])

# fig.show()