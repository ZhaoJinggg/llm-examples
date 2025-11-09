import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage

@st.cache_resource
def firebase_init():
    try:
        service_account_info = dict(st.secrets["firestore"])
        storage_bucket_url = service_account_info.get("storageBucket")
        
        if not storage_bucket_url:
            raise ValueError("storageBucket not found in secrets.")
        
        cred = credentials.Certificate(service_account_info)
            
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"storageBucket": storage_bucket_url})

        db = firestore.client()
        bucket = storage.bucket(storage_bucket_url)
        
        return db, bucket
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Firebase: {e}")