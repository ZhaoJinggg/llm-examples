import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import Config

@st.cache_resource
def get_reranker_model():
    # Reranker Model
    model_name = Config.RERANKER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Auto detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)    
    
    # For float16 precision on GPU
    if device == "cuda":
        model = model.half()
        
    return tokenizer, model, device

def rerank_documents(query: str, docs: list, top_n: int = Config.RERANKER_TOP_N):
    # Return if empty
    if not docs:
        return []

    # 1. Initialize Reranker model
    tokenizer, model, device = get_reranker_model()
    
    # 2. Build [Query, Document Content] pairs
    pairs = [[query, doc.page_content] for doc in docs]
    
    # Batch inference
    batch_size = 32
    all_scores = []
    
    # 3. Model inference
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = tokenizer(
                batch_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(device)
            
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy())
            
    # 4. Associate scores with documents
    import numpy as np
    scores_array = np.array(all_scores)
    top_indices = np.argsort(scores_array)[::-1][:top_n]
    
    # 5. Add scores to top n documents
    result_docs = []
    for idx in top_indices:
        doc = docs[idx]
        doc.metadata["relevance_score"] = float(scores_array[idx])
        result_docs.append(doc)
    
    return result_docs