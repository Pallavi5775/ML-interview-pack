
import weaviate
from weaviate.embedded import EmbeddedOptions
import os

# To use RAG, the appropriate generative-xxx module must be:
# Enabled in Weaviate, and
# Specified in the collection definition.
# Each module is tied to a specific group of LLMs, 
# such as generative-cohere for Cohere models, 
# generative-openai for OpenAI models and generative-google for Google models.



client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(
        additional_env_vars={
            "ENABLE_MODULES": "backup-filesystem,text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai",
            "BACKUP_FILESYSTEM_PATH": "/tmp/backups"
        }
    )
    # Add additional options here (see Python client docs for syntax)
)

client.connect()  # Call `connect()` to connect to the server when you use `WeaviateClient`

response = client.get_meta()
print(response)


# Uncomment the next line to exit the Embedded Weaviate server.
client.close()