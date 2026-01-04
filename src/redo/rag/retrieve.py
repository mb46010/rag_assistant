"""Module for retrieving policy documents from a FAISS vector database."""

import os

import faiss
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def retrieve_policies(query: str, faiss_save_path: str = "faiss_index", top_k: int = 5) -> list[NodeWithScore]:
    """Retrieve policies from a FAISS index.

    Args:
        query: Query string.
        faiss_save_path: Path where the FAISS index is saved.
        top_k: Number of policies to retrieve.

    Returns:
        A list of retrieved policies (NodeWithScore objects).
    """
    # 1. Load the FAISS index
    vector_store = FaissVectorStore.from_persist_dir(faiss_save_path)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=faiss_save_path)

    # We need to specify the same embedding model used during ingestion
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

    # 2. Use the index to retrieve the top_k most similar policies
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    return nodes


if __name__ == "__main__":
    # Quick test
    import dotenv

    from redo.rag.ingest import ingest_policies

    dotenv.load_dotenv()

    # Ensure index exists
    if not os.path.exists("faiss_index"):
        print("Creating index...")
        ingest_policies()

    print("Retrieving policies for: 'remote work'")
    results = retrieve_policies("remote work", top_k=2)
    for i, node in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Score: {node.score}")
        print(f"Content: {node.node.get_content()[:200]}...")
