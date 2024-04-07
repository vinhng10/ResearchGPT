from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

client = QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="store")

pipeline = IngestionPipeline(
    transformations=[
        CodeSplitter(
            language="python",
            chunk_lines=40,
            chunk_lines_overlap=15,
            max_chars=1500,
        ),
        OpenAIEmbedding(),
    ],
    vector_store=vector_store,
)

documents = SimpleDirectoryReader(
    input_dir="./agent", exclude=["node_modules", "public"], recursive=True
).load_data()

# Ingest directly into a vector db
nodes = pipeline.run(documents=documents, num_workers=1)

# Create your index
index = VectorStoreIndex.from_vector_store(vector_store)
index.storage_context.persist(persist_dir=".cache")

# storage_context = StorageContext.from_defaults(persist_dir=".cache")
# index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("What is the function to run a read content of a file?")
print(response)
