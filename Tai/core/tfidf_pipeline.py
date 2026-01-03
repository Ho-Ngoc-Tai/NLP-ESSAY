import requests
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from .nlp_utils import vi_tokenizer

# ==============================
# LOAD REMOTE STOPWORDS
# ==============================
def load_stopwords(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        words = []
        for line in response.text.splitlines():
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            words.extend(line.split())
        return list(set(words))
    except:
        return []

STOPWORDS_EN_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
STOPWORDS_VI_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-vi/master/stopwords-vi.txt"
STOPWORDS_ALL = load_stopwords(STOPWORDS_EN_URL) + load_stopwords(STOPWORDS_VI_URL)

# ==============================
# LOGISTIC REGRESSION (PSEUDO-LABEL)
# ==============================
def run_logistic_classifier(tfidf, pseudo_labels):
    # Train Logistic Regression with pseudo-labels (first 50% = important)
    X = tfidf.toarray()
    y = np.array([1 if i in pseudo_labels else 0 for i in range(len(X))])
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    return clf.predict_proba(X)[:, 1]

# ==============================
# TF-IDF + PAGERANK PIPELINE
# ==============================
def run_tfidf(sentences, ratio=0.33, damping=0.85):
    """
    Pipeline A: TF-IDF + PageRank + Logistic Regression
    Steps: TF-IDF vectorization → Cosine similarity → Graph → PageRank → Top-K selection
    """
    # TF-IDF vectorization with n-grams and stopwords removal
    vectorizer = TfidfVectorizer(
        tokenizer=vi_tokenizer,
        ngram_range=(1, 2),  # Unigrams + Bigrams
        stop_words=STOPWORDS_ALL if STOPWORDS_ALL else None
    )
    tfidf = vectorizer.fit_transform(sentences)
    
    # Cosine similarity matrix (TF-IDF is L2-normalized)
    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)
    
    # Build sentence graph and apply PageRank
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, alpha=damping)
    
    # Select top-K sentences
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = max(1, int(len(sentences) * ratio))
    selected_ids = {i for i, _ in ranked[:top_k]}
    
    return {
        "tfidf": tfidf,
        "sim_matrix": sim_matrix,
        "graph": graph,
        "scores": scores,
        "top_k": top_k,
        "selected_ids": selected_ids,
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out(),
        "idf_values": vectorizer.idf_,
    }
