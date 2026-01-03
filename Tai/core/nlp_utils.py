import re
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Try to import underthesea, fallback to simple tokenizer
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

# ==============================
# SENTENCE SPLITTING
# ==============================
S_TAG_PATTERN = re.compile(r'<s\s+docid="[^"]+"\s+num="\d+"\s+wdcount="\d+"">.*?</s>', re.DOTALL)

def split_sentences(text):
    # Handle XML format (DUC dataset) or plain text
    if S_TAG_PATTERN.search(text):
        sentences = re.findall(r'<s[^>]*>(.*?)</s>', text, flags=re.DOTALL)
    else:
        # Clean text
        text = text.replace("\r", "")
        
        # Split by newlines first, then by sentence delimiters
        # Improved regex to handle quotes and various punctuation
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"])|(?<=\.)\s*\n+', text)
    
    return [s.strip() for s in sentences if s.strip()]

# ==============================
# TOKENIZATION
# ==============================
def vi_tokenizer(sentence):
    # Use underthesea if available, else simple split
    if HAS_UNDERTHESEA:
        try:
            tokens = word_tokenize(sentence, format="text").split()
            return [t.lower() for t in tokens if t.isalnum() or "_" in t]
        except:
            pass
    return [t.lower() for t in sentence.split() if t.isalnum()]

# ==============================
# EXPORT GRAPH
# ==============================
def export_graph(graph, scores, png_path=None, gexf_path=None, title="Sentence Graph", with_labels=True):
    # Add PageRank scores to nodes
    for node in graph.nodes:
        graph.nodes[node]["pagerank"] = scores.get(node, 0.0)
    
    if gexf_path:
        nx.write_gexf(graph, gexf_path)
    
    if png_path:
        # Check if graph has nodes
        if graph.number_of_nodes() == 0:
            # Create empty plot with message
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'No graph to display\n(empty or single node)', 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph, seed=42, k=0.5)
        
        pr_values = np.array([scores.get(n, 0.0) for n in graph.nodes])
        pr_norm = pr_values / pr_values.max() if len(pr_values) > 0 and pr_values.max() > 0 else pr_values
        
        nodes = nx.draw_networkx_nodes(graph, pos, node_size=1000, 
                                       node_color=pr_norm, cmap=plt.get_cmap("YlOrRd"), alpha=0.9)
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=1.5)
        
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
    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity Score')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Index')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
