"""
Quick test script for Pipeline C without browser
"""
import sys
sys.path.insert(0, '.')

from core.multi_doc_ranking import run_multi_doc_ranking

# Test với 1 văn bản
documents = [
    "Trí tuệ nhân tạo đang thay đổi thế giới. ChatGPT rất thành công. AI giúp tự động hóa công việc."
]

print("=" * 60)
print("TEST PIPELINE C - Multi-Document Ranking")
print("=" * 60)
print(f"\nSố văn bản: {len(documents)}")
print(f"\nVăn bản 1: {documents[0][:50]}...")

try:
    result = run_multi_doc_ranking(documents)
    print("\n✅ SUCCESS! Pipeline C đã chạy được!")
    print(f"\nTop documents: {result['top_docs']}")
    print(f"\nSummary: {result['top_doc_summary']}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
