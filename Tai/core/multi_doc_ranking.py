import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .tfidf_pipeline import STOPWORDS_ALL
from .nlp_utils import split_sentences

# ==============================
# MULTI-DOCUMENT RANKING PIPELINE
# ==============================
def run_multi_doc_ranking(documents, doc_names=None, threshold=0.25, damping=0.85, top_n=3):
    """
    Pipeline C: Multi-Document Ranking
    Two-level: 1) Rank documents by PageRank, 2) Summarize top document
    """
    # Default document names
    if doc_names is None:
        doc_names = [f"doc_{i+1:03d}" for i in range(len(documents))]
    
    # Clean all documents (remove HTML/XML tags)
    def clean(text):
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^A-Za-z0-9\.,!?\s'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    
    cleaned_docs = [clean(doc) for doc in documents]
    
    # TF-IDF vectorization for documents
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS_ALL if STOPWORDS_ALL else None,
        max_features=6000
    )
    doc_tfidf = vectorizer.fit_transform(cleaned_docs)
    
    # Compute document similarity and build graph
    doc_sim_matrix = cosine_similarity(doc_tfidf)
    np.fill_diagonal(doc_sim_matrix, 0)
    
    N = len(documents)
    doc_graph = nx.Graph()
    for i in range(N):
        for j in range(i+1, N):
            if doc_sim_matrix[i][j] > threshold:
                doc_graph.add_edge(i, j, weight=doc_sim_matrix[i][j])
    
    # Apply PageRank on document graph
    if doc_graph.number_of_nodes() > 0:
        doc_scores = nx.pagerank(doc_graph, alpha=damping)
    else:
        doc_scores = {i: 1.0/N for i in range(N)}
    
    # Select top-N documents
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [i for i, _ in ranked_docs[:top_n]]
    
    # Summarize the top-1 document using sentence-level PageRank
    top_doc_text = cleaned_docs[top_docs[0]]
    sentences = split_sentences(top_doc_text)
    
    if len(sentences) < 2:
        top_doc_summary = sentences
    else:
        sent_vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ALL if STOPWORDS_ALL else None)
        sent_tfidf = sent_vectorizer.fit_transform(sentences)
        sent_sim = cosine_similarity(sent_tfidf)
        np.fill_diagonal(sent_sim, 0)
        
        sent_graph = nx.from_numpy_array(sent_sim)
        sent_scores = nx.pagerank(sent_graph, alpha=damping)
        
        top_k = max(2, min(3, int(len(sentences) * 0.33)))
        ranked_sents = sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)
        selected_ids = sorted([i for i, _ in ranked_sents[:top_k]])
        top_doc_summary = [sentences[i] for i in selected_ids]
    
    return {
        "cleaned_docs": cleaned_docs,
        "doc_names": doc_names,
        "doc_tfidf": doc_tfidf,
        "doc_sim_matrix": doc_sim_matrix,
        "doc_graph": doc_graph,
        "doc_scores": doc_scores,
        "top_docs": top_docs,
        "top_doc_summary": top_doc_summary,
    }
