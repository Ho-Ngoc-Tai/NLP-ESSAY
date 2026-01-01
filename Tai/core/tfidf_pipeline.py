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
    """
    Load stopwords from remote URL.
    
    Args:
        url: URL to stopwords file
        
    Returns:
        List of stopwords
    """
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
        print(f"Warning: Could not load stopwords from {url}")
        return []

STOPWORDS_EN_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
STOPWORDS_VI_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-vi/master/stopwords-vi.txt"
STOPWORDS_ALL = load_stopwords(STOPWORDS_EN_URL) + load_stopwords(STOPWORDS_VI_URL)

# ==============================
# LOGISTIC REGRESSION (PSEUDO-LABEL)
# ==============================
def run_logistic_classifier(tfidf, pseudo_labels):
    """
    Train a Logistic Regression classifier using pseudo-labels.
    
    Pseudo-labeling strategy: First 50% of sentences are labeled as important.
    This is a simple heuristic assuming important information appears early.
    
    Args:
        tfidf: TF-IDF sentence matrix (sparse)
        pseudo_labels: Set of sentence indices labeled as important
        
    Returns:
        probs: Probability of each sentence being important (numpy array)
    """
    # Feature matrix, each row corresponds to one sentence
    X = tfidf.toarray()
    
    # Pseudo-label assignment: label = 1 → important, label = 0 → not important
    y = np.array([
        1 if i in pseudo_labels else 0
        for i in range(len(X))
    ])
    
    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    # Predict probability of importance for each sentence
    probs = clf.predict_proba(X)[:, 1]
    
    return probs

# ==============================
# TF-IDF + PAGERANK PIPELINE
# ==============================
def run_tfidf(sentences, ratio=0.33, damping=0.85):
    """
    Pipeline A: TF-IDF + PageRank + Logistic Regression
    
    Steps:
    1. Convert sentences into TF-IDF vectors (with n-grams and stopwords removal)
    2. Compute cosine similarity between sentence vectors
    3. Build a sentence similarity graph
    4. Apply PageRank to estimate sentence importance
    5. Select top-K sentences as summary candidates
    
    Args:
        sentences: List of sentence strings
        ratio: Ratio of sentences to select (default: 0.33 = 33%)
        damping: PageRank damping factor (default: 0.85)
        
    Returns:
        Dictionary containing:
        - tfidf: TF-IDF matrix
        - sim_matrix: Cosine similarity matrix
        - graph: NetworkX graph
        - scores: PageRank scores
        - top_k: Number of selected sentences
        - selected_ids: Selected sentence indices
        - vectorizer: Fitted TF-IDF vectorizer
        - feature_names: Vocabulary terms
        - idf_values: IDF values
    """
    # TF-IDF vectorization with n-grams and stopwords removal
    vectorizer = TfidfVectorizer(
        tokenizer=vi_tokenizer,
        ngram_range=(1, 2),  # Unigrams + Bigrams
        stop_words=STOPWORDS_ALL if STOPWORDS_ALL else None
    )
    tfidf = vectorizer.fit_transform(sentences)
    
    # Cosine similarity matrix
    # TF-IDF is L2-normalized, so dot product = cosine similarity
    sim_matrix = cosine_similarity(tfidf)
    
    # Remove self-similarity (diagonal = 0)
    np.fill_diagonal(sim_matrix, 0)
    
    # Build sentence similarity graph
    # Nodes: sentences, Edges: cosine similarity weights
    graph = nx.from_numpy_array(sim_matrix)
    
    # Apply PageRank on the sentence graph
    # PageRank captures sentence importance from global graph structure
    scores = nx.pagerank(graph, alpha=damping)
    
    # Sort sentences by PageRank score (descending)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top-K sentences for summary
    top_k = max(1, int(len(sentences) * ratio))
    
    # Sentence indices selected by PageRank
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
