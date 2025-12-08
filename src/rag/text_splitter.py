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
    
    doc_splits = text_splitter.split_documents(documents)
    return doc_splits