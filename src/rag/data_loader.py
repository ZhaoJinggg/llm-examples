import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile

def load_documents(uploaded_file: UploadedFile, ref_id: str) -> List[Document]:
    # 1. Initialize temporary file path
    tmp_file_path = None
    
    try:
        # 2. Temporary file to store documents
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 3. Load the documents
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # 4. Add metadata about the source file
        for doc in documents:
            doc.metadata["name"] = uploaded_file.name
            doc.metadata["ref_id"] = ref_id
        
        return documents
    
    finally:
        # 6. Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)