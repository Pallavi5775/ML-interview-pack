import asyncio
from hybrid_rag_langchain_langgraph import ingest_document, init_neo4j, init_weaviate



async def main():
    driver = init_neo4j()
    client = init_weaviate()
    try:
        await ingest_document("./article.txt")
    finally:
        if driver:
            await driver.close()
        if client:
            client.close()
        print("✅ All connections closed.")

if __name__ == "__main__":
    asyncio.run(main())
