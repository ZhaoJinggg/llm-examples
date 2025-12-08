import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from firebase_admin import firestore
from math import ceil
import io
import zipfile
import base64
import time
from src.firebase_init import firebase_init
from src.rag.data_loader import load_documents
from src.rag.text_splitter import chunk_documents
from src.rag.vectorstore import index_documents, delete_documents

# --- Firebase Initialization ---
try:
    db, bucket = firebase_init()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# --- Page Configuration ---
st.title("📝 Knowledge Base")

# --- Functions ---
def ingest_files(uploaded_file, ref_id: str):
    # Load the documents for indexing 
    documents = load_documents(uploaded_file, ref_id)
    st.toast(f"📄 Loaded {len(documents)} pages of documents...", icon='⏳')
    
    # Chunk the documents
    chunks = chunk_documents(documents)
    st.toast(f"✂️ Split {uploaded_file.name} into {len(chunks)} chunks.", icon='⏳') 
         
    # Index these documents 
    ids = index_documents(chunks)
    st.toast(f"📚 Indexed {len(ids)} chunks to Vector Store.", icon='⏳')
    
    return len(ids)
    
def upload_files(uploaded_files):    
    for uploaded_file in uploaded_files:
        st.write(f"Uploading {uploaded_file.name}...")
        try:
            # 1. Upload to Firebase Storage
            file_path = f"knowledge_base/{uploaded_file.name}"
            blob = bucket.blob(file_path)
            
            # Ensure file pointer is at the beginning
            uploaded_file.seek(0)
            blob.upload_from_file(uploaded_file, content_type=uploaded_file.type)

            # 2. Store Metadata in Firestore
            file_metadata = {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "path": file_path,
                "date": firestore.SERVER_TIMESTAMP,
                "file_size": uploaded_file.size
            }
            
            # Add metadata to the "knowledge_base" collection
            update_time, doc_ref = db.collection("knowledge_base").add(file_metadata)

            # 3. Ingest the files
            chunk_count = ingest_files(uploaded_file, ref_id=doc_ref.id) 
            
            # Only show when all steps are successful
            st.success(f"✅ Successfully uploaded {uploaded_file.name}")
            
        except Exception as e:
            # Otherwise, show the error
            st.error(f"❌ Error uploading {uploaded_file.name}: {e}")
    
    fetch_files.clear()
    time.sleep(1)
    # Refresh the page
    st.rerun()

@st.cache_data()
def fetch_files():
    """Fetches files from Firestore and formats metadata."""
    files_list = []
    try:
        files_ref = db.collection("knowledge_base").stream()
        
        for doc in files_ref:
            data = doc.to_dict()
            data['id'] = doc.id
            
            # Format date to "Month Day, Year" format
            dt = data['date'].replace(tzinfo=None) 
            data['date'] = dt.strftime("%b %d, %Y")
            
            # Format file size to KB or MB
            size_bytes = data['file_size']
            if size_bytes >= 1024 * 1024:  # >= 1 MB
                size_mb = size_bytes / (1024 * 1024)
                data['file_size'] = f"{size_mb:.2f} MB"
            elif size_bytes >= 1024:  # >= 1 KB
                size_kb = size_bytes / 1024
                data['file_size'] = f"{size_kb:.2f} KB"
            else:
                data['file_size'] = f"{size_bytes} bytes"
                
            files_list.append(data)
        
        return files_list
    except Exception as e:
        st.error(f"Error fetching files: {e}")
        return []

def download_files(selected_indices, df_paginated):
    """Downloads selected files from Firebase Storage and zips them in memory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for item in selected_indices:
            try:
                path = df_paginated.iloc[item]['path']
                file_name = df_paginated.iloc[item]['name']
                blob = bucket.blob(path)
                if blob.exists():
                    file_bytes = blob.download_as_bytes()
                    zip_file.writestr(file_name, file_bytes)
                else:
                    st.error(f"Could not find '{file_name}' in storage.")
            except Exception as e:
                st.error(f"Error downloading '{file_name}': {e}")
    return zip_buffer.getvalue()

def delete_files(selected_indices, df_paginated):
    """Deletes selected files from Firebase Storage, Firestore, and Pinecone."""
    ref_ids = []
    with st.spinner(text="Deleting documents"):
        for item in selected_indices:
            doc_id = df_paginated.iloc[item]['id']
            path = df_paginated.iloc[item]['path']
            file_name = df_paginated.iloc[item]['name']
            
            try:
                # Delete from Firestore
                db.collection("knowledge_base").document(doc_id).delete()
                
                # Delete from Firebase Storage
                blob = bucket.blob(path)
                blob.delete()

                ref_ids.append(doc_id)
            except Exception as e:
                st.error(f"Error deleting {file_name}: {e}")
        
        if ref_ids:
            try:
                delete_documents(ref_ids)
            except Exception as e:
                st.error(f"Error removing vectors for {len(ref_ids)} document(s): {e}")
    
    st.toast('✔️ The selected documents are deleted.', icon='🎉')
    fetch_files.clear()
    # Reset page if current page becomes empty after deletion
    if "curr_page" in st.session_state:
        st.session_state.curr_page = 1
    st.rerun()

# --- File Upload Logic ---
with st.expander("Upload New Files", expanded=False):
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload your materials",
            type=["pdf"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Upload Files")

if submitted and uploaded_files:
    upload_files(uploaded_files)

# --- File Display Logic ---
st.header("🗂️ Files")
st.caption("Manage documents in your knowledge base. Select rows to download or delete.")

# Fetch the file data
file_data = fetch_files()

if file_data:
    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(file_data)
    
    # Sort by name in ascending order
    df = df.sort_values('name', ascending=True).reset_index(drop=True)
    
    # Define the order of columns
    columns_order = ['name', 'file_size', 'type', 'date', 'id', 'path']
    df = df[columns_order]
    
    # Pagination settings
    page_size = 5
    total_pages = ceil(len(df)/page_size)

    if "curr_page" not in st.session_state:
        st.session_state.curr_page = 1

    curr_page = min(st.session_state['curr_page'], total_pages)

    # Displaying pagination buttons
    if total_pages > 1: 
        prev, next, _, col3 = st.columns([1,1,6,2])

        if next.button("Next"):
            curr_page = min(curr_page + 1, total_pages)
            st.session_state['curr_page'] = curr_page

        if prev.button("Prev"):
            curr_page = max(curr_page - 1, 1)
            st.session_state['curr_page'] = curr_page

        with col3: 
            st.write("Page: ", curr_page, "/", total_pages)

    start_index = (curr_page - 1) * page_size
    end_index = curr_page * page_size
    df_paginated = df.iloc[start_index:end_index]

    # Display the DataFrame
    docs = st.dataframe(
        df_paginated,
        width=2000,
        column_config={
            "id": None, 
            "name": "Name",
            "file_size": "Size",
            "type": "Type",
            "date": "Creation Date",
            "path": None,
        },
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )
    
    # Download and Delete functionality
    selected_docs = docs.selection.rows
    if len(selected_docs) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            download_button = st.button("📥 Download", key="download", use_container_width=True)  
            if download_button:
                with st.spinner("Preparing files for download..."):
                    zip_data = download_files(selected_docs, df_paginated)
                    if zip_data:
                        b64 = base64.b64encode(zip_data).decode()
                        html = f"""
                        <html>
                            <body>
                            <a id="autodownload" href="data:application/zip;base64,{b64}" download="Download.zip"></a>
                            <script>
                                const link = document.getElementById('autodownload');
                                if (link) {{ link.click(); }}
                            </script>
                            </body>
                        </html>
                        """
                        components.html(html, height=0, width=0)
                        st.toast("✔️ The selected documents are downloaded.", icon='🎉')    
        with col2:
            delete_button = st.button("🗑️ Delete", key="delete_docs", use_container_width=True)
            if delete_button:
                delete_files(selected_docs, df_paginated)
else:
    st.info("No files found in the knowledge base. Upload some files to get started.")