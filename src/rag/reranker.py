import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import Config

@st.cache_resource(show_spinner="Loading Reranker Model...")
def get_reranker_model():
    # Reranker Model
    model_name = Config.RERANKER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Auto detect device (GPU > MPS (Mac) > CPU)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    model.to(device)
    return tokenizer, model, device

def rerank_documents(query: str, docs: list, top_n: int = Config.RERANKER_TOP_N):
    # Return if empty
    if not docs:
        return []

    # 1. Initialize Reranker model
    tokenizer, model, device = get_reranker_model()
    
    # 2. Build [Query, Document Content] pairs
    pairs = [[query, doc.page_content] for doc in docs]
    
    # 3. Model inference
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        # Calculate scores
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    # 4. Associate scores with documents
    scored_docs = []
    for i, doc in enumerate(docs):
        score = scores[i].item()
        # Add score to document metadata
        doc.metadata["relevance_score"] = score
        scored_docs.append((doc, score))
    
    # 5. Sort documents by score in descending order (higher score is more relevant)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Return top n documents
    return [doc for doc, score in scored_docs[:top_n]]