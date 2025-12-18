from typing import List
from langchain_core.documents import Document
from src.rag.retriever import hybrid_retriever

def index_documents(chunks: List[Document]) -> None:
    if not chunks:
        return
    
    retriever = hybrid_retriever()
    documents  = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    # Upload Dense + Sparse Upsert
    retriever.add_texts(documents, metadatas=metadatas)

def delete_index(ref_ids: List[str]) -> None:
    # Remove empty values and duplicates
    unique_ids = sorted({ref_id for ref_id in ref_ids if ref_id}) 
    if not unique_ids:
        # Nothing to delete
        return
    
    from pinecone import Pinecone
    from src.config import Config
    
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    
    try:
        # Delete all vectors with the given ref_ids
        index.delete(filter={"ref_id": {"$in": unique_ids}})
    except Exception as exc:
        raise RuntimeError(f"Failed to delete vectors for ref_ids {unique_ids}") from exc