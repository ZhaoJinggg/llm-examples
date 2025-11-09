import io
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vector_store import VectorStoreService

class DocumentProcessor:
    """Service for processing and indexing documents"""
    
    def __init__(self):
        """Initialize document processor"""
        self.vector_store_service = VectorStoreService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            add_start_index=True,
        )
    
    def process_and_index_pdf(self, file_bytes, file_name, file_metadata=None):
        """
        Process a PDF file and index it into vector store
        
        Args:
            file_bytes: PDF file bytes
            file_name: Name of the file
            file_metadata: Additional metadata to attach to chunks
            
        Returns:
            Number of chunks indexed
        """
        try:
            # Create a temporary file to load the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Add custom metadata
            for doc in documents:
                doc.metadata['source_file'] = file_name
                if file_metadata:
                    doc.metadata.update(file_metadata)
            
            # Split documents
            doc_splits = self.text_splitter.split_documents(documents)
            
            # Index chunks into vector store
            self.vector_store_service.add_documents(documents=doc_splits)
            
            # Clean up temp file
            import os
            os.unlink(tmp_file_path)
            
            return len(doc_splits)
            
        except Exception as e:
            raise Exception(f"Error processing PDF {file_name}: {str(e)}")
    
    def process_multiple_pdfs(self, pdf_files_data):
        """
        Process multiple PDF files
        
        Args:
            pdf_files_data: List of tuples (file_bytes, file_name, metadata)
            
        Returns:
            Total number of chunks indexed
        """
        total_chunks = 0
        for file_bytes, file_name, metadata in pdf_files_data:
            chunks = self.process_and_index_pdf(file_bytes, file_name, metadata)
            total_chunks += chunks
        return total_chunks

