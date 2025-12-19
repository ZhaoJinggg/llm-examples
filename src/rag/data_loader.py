import os
import tempfile
from typing import List, Dict, Type
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Mapping of file extensions to their corresponding loaders
LOADER_MAPPING: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}

# Supported file types for external reference
SUPPORTED_EXTENSIONS = list(LOADER_MAPPING.keys())


def get_file_extension(filename: str) -> str:
    """Extract the file extension from a filename."""
    return os.path.splitext(filename)[1].lower()


def load_documents(uploaded_file: UploadedFile, ref_id: str) -> List[Document]:
    # 1. Get file extension and validate
    file_ext = get_file_extension(uploaded_file.name)
    
    if file_ext not in LOADER_MAPPING:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {supported}")
    
    # 2. Initialize temporary file path
    tmp_file_path = None
    
    try:
        # 3. Create temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 4. Get the appropriate loader and load documents
        loader_class = LOADER_MAPPING[file_ext]
        loader = loader_class(tmp_file_path)
        documents = loader.load()
        
        # 5. Add metadata about the source file
        for doc in documents:
            doc.metadata["name"] = uploaded_file.name
            doc.metadata["ref_id"] = ref_id
        
        return documents
    
    finally:
        # 6. Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
