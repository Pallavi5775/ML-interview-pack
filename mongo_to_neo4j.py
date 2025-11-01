import asyncio
from neo4j import AsyncGraphDatabase, GraphDatabase
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm


# ===============================
# CONFIG
# ===============================
MONGO_URI = 'mongodb+srv://pallavidapriya75_db_user:h4bkjpuGqfUaoNbx@cluster0.se84jvl.mongodb.net/?retryWrites=true&w=majority&tls=true'
MONGO_DB = "iiT_data_fetcher"
MONGO_COLLECTION = "market_data"

NEO4J_URI = "neo4j+s://4799003d.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "PGdrQZkl_LcdAk25s4nRiH1Wpx4wS07Duhp6xKUGn5s"

# ===============================
# CONNECT
# ===============================
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB]

from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j+ssc://4799003d.databases.neo4j.io"
AUTH = ("neo4j", "PGdrQZkl_LcdAk25s4nRiH1Wpx4wS07Duhp6xKUGn5s")






# ===============================
# SAFE DATE EXTRACTOR
# ===============================
def extract_date(value):
    """Handles both {'$date': ...} and datetime.datetime"""
    if isinstance(value, dict):
        return value.get("$date")
    elif hasattr(value, "isoformat"):
        return value.isoformat()
    elif isinstance(value, str):
        return value
    return None


# ===============================
# NEO4J INSERT FUNCTION
# ===============================
async def insert_stock(tx, doc):
    symbol = doc.get("symbol")
    exchange = doc.get("exchange")
    currency = doc.get("currency")

    # Safely extract date from doc["datetime"]
    date = extract_date(doc.get("datetime"))
    prices = doc.get("prices", {})
    fundamentals = doc.get("fundamentals", {})
    metadata = doc.get("metadata", {})

    # --- Stock Node ---
    await tx.run("""
        MERGE (s:Stock {symbol: $symbol})
        SET s.exchange = $exchange, s.currency = $currency
    """, symbol=symbol, exchange=exchange, currency=currency)

    # --- Prices ---
    if prices:
        await tx.run("""
            MERGE (p:Price {symbol: $symbol, date: date($date)})
            SET p += $prices
            WITH p MATCH (s:Stock {symbol:$symbol})
            MERGE (s)-[:HAS_PRICE]->(p)
        """, symbol=symbol, date=date[:10] if date else None, prices=prices)

    # --- Fundamentals ---
    if fundamentals:
        await tx.run("""
            MERGE (f:Fundamentals {symbol: $symbol, date: date($date)})
            SET f += $fundamentals
            WITH f MATCH (s:Stock {symbol:$symbol})
            MERGE (s)-[:HAS_FUNDAMENTALS]->(f)
        """, symbol=symbol, date=date[:10] if date else None, fundamentals=fundamentals)

    # --- Metadata / Source ---
    if metadata:
        src = metadata.get("source", "unknown")
        ingested_at = extract_date(metadata.get("ingested_at"))
        await tx.run("""
            MERGE (src:Source {name: $src})
            WITH src MATCH (s:Stock {symbol:$symbol})
            MERGE (s)-[:INGESTED_FROM {ingested_at: datetime($ingested_at)}]->(src)
        """, symbol=symbol, src=src, ingested_at=ingested_at)


# ===============================
# PROCESS RECORD
# ===============================
async def process_record(driver, doc):
    try:
        async with driver.session() as session:
            await session.execute_write(insert_stock, doc)
        print(f"✅ Inserted: {doc.get('symbol')}")
    except Exception as e:
        print(f"⚠️ Error on {doc.get('symbol')}: {e}")


# ===============================
# MAIN
# ===============================
async def main():
    print("🚀 Connecting to Neo4j Aura...")
    driver = AsyncGraphDatabase.driver(URI, auth=AUTH)
    await driver.verify_connectivity()
    print("✅ Connected to Neo4j Aura")

    # ✅ Fetch only first 5 docs
    cursor = db[MONGO_COLLECTION].find({}).limit(5)
    records = [doc async for doc in cursor]
    print(f"📦 Retrieved {len(records)} documents from MongoDB")

    for doc in records:
        await process_record(driver, doc)

    await driver.close()
    mongo_client.close()
    print("🎯 Completed upload of 5 documents!")


if __name__ == "__main__":
    asyncio.run(main())