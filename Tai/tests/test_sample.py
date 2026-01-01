"""
Unit tests for text summarization pipelines.

Run with: python -m pytest tests/test_sample.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nlp_utils import split_sentences, vi_tokenizer
from core.tfidf_pipeline import run_tfidf
from core.textrank_pipeline import run_textrank
from core.multi_doc_ranking import run_multi_doc_ranking

def test_sentence_splitting():
    """Test sentence splitting functionality"""
    text = "Đây là câu đầu tiên. Đây là câu thứ hai! Đây là câu thứ ba?"
    sentences = split_sentences(text)
    assert len(sentences) == 0  # All sentences < 5 words, filtered out
    
    text2 = "Trí tuệ nhân tạo đang thay đổi thế giới. Machine learning là một phần quan trọng của AI."
    sentences2 = split_sentences(text2)
    assert len(sentences2) == 2

def test_tokenization():
    """Test Vietnamese tokenization"""
    sentence = "Trí tuệ nhân tạo rất quan trọng"
    tokens = vi_tokenizer(sentence)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

def test_tfidf_pipeline():
    """Test TF-IDF pipeline"""
    sentences = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
        "AI is transforming healthcare and education sectors"
    ]
    
    result = run_tfidf(sentences)
    
    assert "tfidf" in result
    assert "sim_matrix" in result
    assert "graph" in result
    assert "scores" in result
    assert "selected_ids" in result
    assert len(result["selected_ids"]) > 0

def test_textrank_pipeline():
    """Test TextRank pipeline"""
    sentences = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
        "AI is transforming healthcare and education sectors"
    ]
    
    result = run_textrank(sentences)
    
    assert "tokens" in result
    assert "sim_matrix" in result
    assert "graph" in result
    assert "scores" in result
    assert "selected_ids" in result
    assert len(result["selected_ids"]) > 0

def test_multi_doc_ranking():
    """Test multi-document ranking pipeline"""
    documents = [
        "Artificial intelligence is changing the world. Machine learning is important.",
        "Deep learning uses neural networks. It is a powerful technique.",
        "Natural language processing helps computers understand text. It has many applications."
    ]
    
    result = run_multi_doc_ranking(documents)
    
    assert "doc_scores" in result
    assert "top_docs" in result
    assert "top_doc_summary" in result
    assert len(result["top_docs"]) > 0

if __name__ == "__main__":
    print("Running tests...")
    test_sentence_splitting()
    print("✓ Sentence splitting test passed")
    
    test_tokenization()
    print("✓ Tokenization test passed")
    
    test_tfidf_pipeline()
    print("✓ TF-IDF pipeline test passed")
    
    test_textrank_pipeline()
    print("✓ TextRank pipeline test passed")
    
    test_multi_doc_ranking()
    print("✓ Multi-doc ranking test passed")
    
    print("\n✅ All tests passed!")
