import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Try to import underthesea, fallback to simple tokenizer if not available
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    print("Warning: underthesea not installed. Using simple tokenizer.")

# ==============================
# SENTENCE SPLITTING
# ==============================
S_TAG_PATTERN = re.compile(
    r'<s\s+docid="[^"]+"\s+num="\d+"\s+wdcount="\d+"">.*?</s>',
    re.DOTALL
)

def split_sentences(text):
    """
    Split text into sentences.
    Handles both XML-tagged format and plain text.
    
    Args:
        text: Input text string
        
    Returns:
        List of sentence strings (filtered: > 5 words)
    """
    if S_TAG_PATTERN.search(text):
        # XML format (DUC dataset)
        sentences = re.findall(r'<s[^>]*>(.*?)</s>', text, flags=re.DOTALL)
    else:
        # Plain text
        text = text.replace("\r", "")
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 0]
        
        sentences = []
        for line in lines:
            # Split by sentence delimiters
            parts = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend(parts)
    
    # Filter short sentences
    return [s.strip() for s in sentences if len(s.split()) > 5]

# ==============================
# TOKENIZATION
# ==============================
def vi_tokenizer(sentence):
    """
    Tokenize Vietnamese sentence using underthesea.
    Falls back to simple split if underthesea is not available.
    
    Args:
        sentence: Input sentence string
        
    Returns:
        List of lowercase tokens
    """
    if HAS_UNDERTHESEA:
        try:
            tokens = word_tokenize(sentence, format="text").split()
            return [t.lower() for t in tokens if t.isalnum() or "_" in t]
        except:
            pass
    
    # Fallback to simple tokenization
    return [t.lower() for t in sentence.split() if t.isalnum()]

# ==============================
# EXPORT GRAPH
# ==============================
def export_graph(graph, scores, png_path=None, gexf_path=None, title="Sentence Graph", with_labels=True):
    """
    Export sentence graph for visualization and analysis.
    
    Args:
        graph: networkx graph
        scores: PageRank scores (dict)
        png_path: path to save PNG (optional)
        gexf_path: path to save GEXF for Gephi (optional)
        title: Graph title
        with_labels: Whether to show node labels
    """
    # Add PageRank scores to nodes
    for node in graph.nodes:
        graph.nodes[node]["pagerank"] = scores.get(node, 0.0)
    
    # Export GEXF
    if gexf_path:
        nx.write_gexf(graph, gexf_path)
    
    # Export PNG
    if png_path:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph, seed=42, k=0.5)
        
        pr_values = np.array([scores.get(n, 0.0) for n in graph.nodes])
        if pr_values.max() > 0:
            pr_norm = pr_values / pr_values.max()
        else:
            pr_norm = pr_values
        
        # Draw nodes with color based on PageRank
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=1000,
            node_color=pr_norm,
            cmap=plt.get_cmap("YlOrRd"),
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.3,
            width=1.5
        )
        
        # Draw labels
        if with_labels:
            labels = {i: f"S{i+1}" for i in graph.nodes}
            nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_weight='bold')
        
        plt.colorbar(nodes, label="PageRank Score")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

def save_heatmap(sim_matrix, path, title="Cosine Similarity Heatmap"):
    """
    Save similarity matrix as heatmap.
    
    Args:
        sim_matrix: NxN similarity matrix
        path: Output path
        title: Heatmap title
    """
    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity Score')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Index')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
