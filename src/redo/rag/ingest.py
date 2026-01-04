"""Module for ingesting policy documents into a FAISS vector database."""

import os

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def ingest_policies(
    policies_dir: str = "policies", chunk_size: int = 500, faiss_save_path: str = "faiss_index"
) -> VectorStoreIndex:
    """Load policies from markdown files into a FAISS vector database.

    Args:
        policies_dir: Directory containing policy markdown files.
        chunk_size: Number of tokens per chunk.
        faiss_save_path: Path to save the FAISS index.

    Returns:
        A LlamaIndex VectorStoreIndex.
    """
    # 1. Load the policies (.md)
    reader = SimpleDirectoryReader(
        input_dir=policies_dir,
        required_exts=[".md"],
    )
    documents = reader.load_data()

    # 2. Split documents into chunks
    splitter = SentenceSplitter(chunk_size=chunk_size)

    # 3. Create OpenAI embeddings (large-3)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # 4. Create FAISS vector store
    # Register FAISS index
    # We use a simple flat index for CPU
    d = 3072  # Dimension for text-embedding-3-large
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # 5. Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )

    # 6. Persist the index
    if faiss_save_path:
        storage_context.persist(persist_dir=faiss_save_path)

    return index


if __name__ == "__main__":
    # Quick test
    import dotenv

    dotenv.load_dotenv()
    index = ingest_policies()
    print(f"Ingested {len(index.ref_doc_info)} documents.")
