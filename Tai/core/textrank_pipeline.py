import numpy as np
import networkx as nx
from .nlp_utils import vi_tokenizer

# ==============================
# OVERLAP SIMILARITY
# ==============================
def overlap_similarity(tokens_i, tokens_j):
    """
    Compute lexical overlap-based similarity between two sentences.
    This is the classical TextRank similarity (non-vector-based).
    
    Formula: sim(Si, Sj) = |Si ∩ Sj| / (log|Si| + log|Sj|)
    
    Args:
        tokens_i: List of tokens from sentence i
        tokens_j: List of tokens from sentence j
        
    Returns:
        Similarity score (float)
    """
    # If either sentence is empty after tokenization, similarity is zero
    if not tokens_i or not tokens_j:
        return 0.0
    
    # Convert token lists to sets to remove duplicates
    set_i = set(tokens_i)
    set_j = set(tokens_j)
    
    # Intersection of tokens between two sentences
    overlap = set_i & set_j
    
    # If there is no common token, similarity is zero
    if not overlap:
        return 0.0
    
    # Overlap-based similarity formula
    # Normalization prevents longer sentences from dominating
    return len(overlap) / (np.log(len(set_i)) + np.log(len(set_j)))

# ==============================
# OVERLAP-BASED TEXTRANK PIPELINE
# ==============================
def run_textrank(sentences, ratio=0.33, damping=0.85):
    """
    Pipeline B: TextRank (Overlap-based)
    
    Steps:
    1. Tokenize sentences (no explicit stopword removal)
    2. Compute overlap-based sentence similarity
    3. Build sentence graph using similarity matrix
    4. Apply PageRank to rank sentence importance
    5. Select top-K sentences for summary
    
    Note: Stopwords are not explicitly removed in this overlap-based TextRank.
    This may increase lexical overlap between sentences and make behavior closer
    to TF-IDF-based PageRank for texts with homogeneous vocabulary.
    
    Args:
        sentences: List of sentence strings
        ratio: Ratio of sentences to select (default: 0.33 = 33%)
        damping: PageRank damping factor (default: 0.85)
        
    Returns:
        Dictionary containing:
        - tokens: Tokenized sentences
        - sim_matrix: Overlap-based similarity matrix
        - graph: NetworkX graph
        - scores: PageRank scores
        - top_k: Number of selected sentences
        - selected_ids: Selected sentence indices
    """
    # Tokenize all sentences
    tokenized = [vi_tokenizer(s) for s in sentences]
    
    N = len(sentences)
    
    # Initialize similarity matrix
    sim_matrix = np.zeros((N, N))
    
    # Compute pairwise overlap similarity
    for i in range(N):
        for j in range(N):
            # Self-similarity is explicitly ignored (no self-loops)
            if i != j:
                sim_matrix[i][j] = overlap_similarity(
                    tokenized[i], tokenized[j]
                )
    
    # Build sentence similarity graph
    # Nodes represent sentences
    # Edge weights represent overlap similarity
    graph = nx.from_numpy_array(sim_matrix)
    
    # Apply PageRank on overlap-based sentence graph
    # PageRank captures sentence importance from lexical overlap structure
    scores = nx.pagerank(graph, alpha=damping)
    
    # Sort sentences by PageRank score (descending)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top-K sentences for summary
    top_k = max(1, int(len(sentences) * ratio))
    
    # Sentence indices selected by TextRank
    selected_ids = {i for i, _ in ranked[:top_k]}
    
    return {
        "tokens": tokenized,
        "sim_matrix": sim_matrix,
        "graph": graph,
        "scores": scores,
        "top_k": top_k,
        "selected_ids": selected_ids,
    }
