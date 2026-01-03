import numpy as np
import networkx as nx
from .nlp_utils import vi_tokenizer

# ==============================
# OVERLAP SIMILARITY
# ==============================
def overlap_similarity(tokens_i, tokens_j):
    # Compute lexical overlap: |Si ∩ Sj| / (log|Si| + log|Sj|)
    if not tokens_i or not tokens_j:
        return 0.0
    
    set_i, set_j = set(tokens_i), set(tokens_j)
    overlap = set_i & set_j
    
    if not overlap:
        return 0.0
    
    return len(overlap) / (np.log(len(set_i)) + np.log(len(set_j)))

# ==============================
# OVERLAP-BASED TEXTRANK PIPELINE
# ==============================
def run_textrank(sentences, ratio=0.33, damping=0.85):
    """
    Pipeline B: TextRank (Overlap-based)
    Steps: Tokenization → Overlap similarity → Graph → PageRank → Top-K selection
    """
    # Tokenize all sentences
    tokenized = [vi_tokenizer(s) for s in sentences]
    N = len(sentences)
    
    # Compute pairwise overlap similarity
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                sim_matrix[i][j] = overlap_similarity(tokenized[i], tokenized[j])
    
    # Build graph and apply PageRank
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, alpha=damping)
    
    # Select top-K sentences
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = max(1, int(len(sentences) * ratio))
    selected_ids = {i for i, _ in ranked[:top_k]}
    
    return {
        "tokens": tokenized,
        "sim_matrix": sim_matrix,
        "graph": graph,
        "scores": scores,
        "top_k": top_k,
        "selected_ids": selected_ids,
    }
