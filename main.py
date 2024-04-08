from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import embeddings
from qdrant_client import QdrantClient


WORKERS = 1
LANGUAGES = ["html", "css", "typescript", "json", "markdown"]
LANGUAGE_MAPPING = {
    ".html": "html",
    ".css": "css",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".md": "markdown",
}

if __name__ == "__main__":
    embedding = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
    client = QdrantClient(path=".cache")
    vector_store = QdrantVectorStore(client=client, collection_name="store")

    # pipelines = {
    #     language: IngestionPipeline(
    #         transformations=[
    #             CodeSplitter(
    #                 language=language,
    #                 chunk_lines=40,
    #                 chunk_lines_overlap=15,
    #                 max_chars=1500,
    #             ),
    #             embedding,
    #             # OpenAIEmbedding(),
    #         ],
    #         vector_store=vector_store,
    #     )
    #     for language in LANGUAGES
    # }

    # reader = SimpleDirectoryReader(
    #     input_dir="./app",
    #     exclude=["node_modules", "public", "__pycache__", "*lock.json", "*.svg"],
    #     recursive=True,
    # )

    # documents = []
    # for document_list in reader.iter_data():
    #     # Ingest directly into a vector db
    #     for document in document_list:
    #         language = LANGUAGE_MAPPING[Path(document.metadata["file_name"]).suffix]
    #         pipeline = pipelines[language]
    #         pipeline.run(documents=[document], num_workers=WORKERS)

    # Create your index
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding)
    # index.storage_context.persist(persist_dir=".cache")

    # storage_context = StorageContext.from_defaults(persist_dir=".cache")
    # index = load_index_from_storage(storage_context, embed_model=embedding)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the css style of App header?")
    print(response)
