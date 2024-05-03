from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
    StorageContext,
    ChatPromptTemplate,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index_client import ChatMessage
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams


Settings.llm = OpenAI(
    api_base="https://development-llm.dbinno.com/v1", temperature=0.1, max_tokens=1024
)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
client = qdrant_client.QdrantClient(
    url="https://development-qdrant.dbinno.com", port=443
)

# client.create_collection(
#     collection_name="test_collection",
#     vectors_config=VectorParams(size=4, distance=Distance.DOT),
# )

documents = SimpleDirectoryReader("documents/").load_data(show_progress=True)
vector_store = QdrantVectorStore(client=client, collection_name="One")
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# _ = VectorStoreIndex.from_documents(
#     documents,
#     insert_batch_size=16,
#     transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=64)],
#     storage_context=storage_context,
#     show_progress=True,
# )
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# index.storage_context.persist()
# storage_context = StorageContext.from_defaults(persist_dir="storage")
# index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=2,
)
response = query_engine.query("How to change my profile picture on iOS and Android?")
response.print_response_stream()
