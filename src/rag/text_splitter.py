from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

def chunk_documents(documents: List[Document]) -> List[Document]:
    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=Config.TEXT_SPLITTER_CHUNK_OVERLAP,
        add_start_index=Config.TEXT_SPLITTER_ADD_START_INDEX,
    )
    
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(documents)
    
    # Ensure each chunk has the ref_id metadata
    for chunk in doc_splits:
        if "ref_id" not in chunk.metadata:
            raise ValueError("Chunk metadata missing ref_id")
        
    # Return the chunks
    return doc_splits