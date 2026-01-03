import os
from collections import Counter
from flask import Flask, render_template, request, jsonify
from core.nlp_utils import split_sentences, export_graph, save_heatmap
from core.tfidf_pipeline import run_tfidf, run_logistic_classifier
from core.textrank_pipeline import run_textrank
from core.multi_doc_ranking import run_multi_doc_ranking

app = Flask(__name__)

OUTPUT_DIR = "static/uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# PIPELINE A - TF-IDF
# ==============================
def pipelineA(text):
    """Process text using TF-IDF + PageRank + Logistic Regression"""
    sentences = split_sentences(text)
    
    if len(sentences) < 1:
        return {
            "sentences": [],
            "sentence_rows": [],
            "summary": [],
            "summary_lr": [],
            "top_k": 0,
            "error": "No valid sentences found"
        }
    
    result = run_tfidf(sentences)
    
    top_k = result["top_k"]
    tfidf = result["tfidf"]
    sim_matrix = result["sim_matrix"]
    pagerank = result["scores"]
    selected_ids = result["selected_ids"]
    feature_names = result["feature_names"]
    idf_values = result["idf_values"]
    analyzer = result["vectorizer"].build_analyzer()
    
    # Pseudo-labels for Logistic Regression
    pseudo_labels = {
        i for i in range(len(sentences))
        if i < len(sentences) // 2  # First half = important
    }
    clf_probs = run_logistic_classifier(tfidf, pseudo_labels)
    lr_ranked = sorted(enumerate(clf_probs), key=lambda x: x[1], reverse=True)
    lr_selected_ids = {i for i, _ in lr_ranked[:top_k]}
    summary_lr = [sentences[i] for i in sorted(lr_selected_ids)]
    
    # Export visualizations
    export_graph(
        result["graph"],
        result["scores"],
        png_path=f"{OUTPUT_DIR}/graphA.png",
        title="TF-IDF Sentence Graph"
    )
    save_heatmap(
        sim_matrix,
        f"{OUTPUT_DIR}/heatmapA.png",
        title="Cosine Similarity Heatmap"
    )
    
    # Build detailed sentence rows for UI
    N = len(sentences)
    d = 0.85
    base_score = round((1 - d) / N, 6)
    
    sentence_rows = []
    for i, s in enumerate(sentences):
        row_vector = tfidf[i].toarray()[0]
        
        # Tokenization
        terms = analyzer(s)
        term_counts = Counter(terms)
        total_terms = len(terms)
        
        # TF-IDF details
        tfidf_rows = []
        for idx, val in enumerate(row_vector):
            if val > 0:
                term = feature_names[idx]
                f_td = term_counts.get(term, 0)
                tf_raw = f_td / total_terms if total_terms > 0 else 0
                tfidf_rows.append({
                    "term": term,
                    "tf_raw": round(tf_raw, 4),
                    "tf": round(val / idf_values[idx], 4),
                    "idf": round(idf_values[idx], 4),
                    "tfidf": round(val, 4)
                })
        
        all_terms = [
            (feature_names[idx], round(row_vector[idx], 4))
            for idx, val in enumerate(row_vector) if val > 0
        ]
        top_terms = sorted(all_terms, key=lambda x: x[1], reverse=True)[:5]
        
        # Cosine similarity details
        cosine_rows = []
        for j, sim in enumerate(sim_matrix[i]):
            if i != j and sim > 0:
                terms_i = dict(all_terms)
                terms_j = dict(
                    (feature_names[idx], round(tfidf[j].toarray()[0][idx], 4))
                    for idx, val in enumerate(tfidf[j].toarray()[0]) if val > 0
                )
                
                shared = []
                for t in set(terms_i.keys()) & set(terms_j.keys()):
                    w1 = terms_i[t]
                    w2 = terms_j[t]
                    shared.append({
                        "term": t,
                        "s1": w1,
                        "s2": w2,
                        "product": round(w1 * w2, 4)
                    })
                
                cosine_rows.append({
                    "pair": f"S{i + 1} - S{j + 1}",
                    "similarity": round(sim, 4),
                    "shared_terms": shared
                })
        
        # PageRank details
        pr_rows = []
        total_contrib = 0.0
        for j in range(N):
            if sim_matrix[j][i] > 0 and j != i:
                w_ji = sim_matrix[j][i]
                sum_w_j = sim_matrix[j].sum()
                contrib = d * (w_ji / sum_w_j) * pagerank[j]
                pr_rows.append({
                    "from": f"S{j + 1}",
                    "w_ji": round(w_ji, 4),
                    "sum_w_j": round(sum_w_j, 4),
                    "pr_j": round(pagerank[j], 4),
                    "contrib": round(contrib, 4),
                })
                total_contrib += contrib
        
        sentence_rows.append({
            "no": i + 1,
            "sentence": s,
            "tfidf_terms": ", ".join([t for t, _ in top_terms]),
            "connections": len(cosine_rows),
            "pagerank": round(pagerank.get(i, 0), 4),
            "lr_score": round(float(clf_probs[i]), 4),
            "selected": "Yes" if i in selected_ids else "No",
            "detail": {
                "tokens": terms,
                "tfidf_rows": tfidf_rows,
                "top_terms": top_terms,
                "cosine_rows": cosine_rows,
                "pagerank_rows": pr_rows,
                "pagerank_base": round(base_score, 4),
                "pagerank_sum": round(total_contrib, 4),
                "pagerank_final": round(base_score + total_contrib, 4),
            }
        })
    
    return {
        "sentences": sentences,
        "sentence_rows": sentence_rows,
        "summary": [sentences[i] for i in sorted(result["selected_ids"])],
        "summary_lr": summary_lr,
        "top_k": top_k,
    }

# ==============================
# PIPELINE B - TEXTRANK
# ==============================
def pipelineB(text):
    """Process text using TextRank (overlap-based)"""
    sentences = split_sentences(text)
    
    if len(sentences) < 1:
        return {
            "sentences": [],
            "sentence_rows": [],
            "summary": [],
            "error": "No valid sentences found"
        }
    
    result = run_textrank(sentences)
    
    export_graph(
        result["graph"],
        result["scores"],
        png_path=f"{OUTPUT_DIR}/graphB.png",
        title="TextRank Sentence Graph"
    )
    
    sentence_rows = []
    for i, s in enumerate(sentences):
        sentence_rows.append({
            "no": i + 1,
            "sentence": s,
            "tokens": result["tokens"][i],
            "pagerank": round(result["scores"].get(i, 0), 4),
            "selected": "Yes" if i in result["selected_ids"] else "No"
        })
    
    return {
        "sentences": sentences,
        "sentence_rows": sentence_rows,
        "summary": [sentences[i] for i in sorted(result["selected_ids"])],
    }

# ==============================
# PIPELINE C - MULTI-DOCUMENT
# ==============================
def pipelineC(documents, doc_names=None):
    """Process multiple documents using multi-document ranking"""
    if len(documents) < 1:
        return {
            "error": "No documents provided"
        }
    
    result = run_multi_doc_ranking(documents, doc_names)
    
    # Export document graph
    export_graph(
        result["doc_graph"],
        result["doc_scores"],
        png_path=f"{OUTPUT_DIR}/graphC.png",
        title="Multi-Document Graph"
    )
    
    # Build document rows
    doc_rows = []
    for i, name in enumerate(result["doc_names"]):
        doc_rows.append({
            "no": i + 1,
            "name": name,
            "pagerank": round(result["doc_scores"].get(i, 0), 4),
            "selected": "Yes" if i in result["top_docs"] else "No"
        })
    
    return {
        "doc_rows": doc_rows,
        "top_docs": result["top_docs"],
        "top_doc_names": [result["doc_names"][i] for i in result["top_docs"]],
        "summary": result["top_doc_summary"],
    }

@app.route("/", methods=["GET", "POST"])
def index():
    context = {}
    
    if request.method == "POST":
        text = request.form.get("text", "")
        pipeline = request.form.get("pipeline", "A")
        
        context["text"] = text
        context["active_pipeline"] = pipeline
        
        if pipeline == "B":
            context["pipelineB"] = pipelineB(text)
        elif pipeline == "C":
            # For demo, split text into "documents" by double newline
            documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
            context["pipelineC"] = pipelineC(documents)
        else:  # Pipeline A
            pa = pipelineA(text)
            context["pipelineA"] = pa
            if "error" not in pa:
                context["sentences"] = pa.get("sentences")
                context["sentence_rows"] = pa.get("sentence_rows")
                context["summary"] = pa.get("summary")
                context["summary_lr"] = pa.get("summary_lr")
                context["top_k"] = pa.get("top_k")
    
    return render_template("index.html", **context)

if __name__ == "__main__":
    print("=" * 60)
    print(">> Text Summarization System")
    print("=" * 60)
    print("3 Pipelines Available:")
    print("  A: TF-IDF + PageRank + Logistic Regression")
    print("  B: TextRank (Overlap-based)")
    print("  C: Multi-Document Ranking")
    print("=" * 60)
    print("Open browser: http://127.0.0.1:5000/")
    print("=" * 60)
    app.run(debug=True)
