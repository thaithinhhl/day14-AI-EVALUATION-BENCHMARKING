from typing import List, Dict, Any

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_case(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> Dict[str, float]:
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        return {"hit_rate": hit_rate, "mrr": mrr}

    async def evaluate_batch(self, dataset: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Mỗi sample cần có:
        - expected_retrieval_ids: List[str]
        - retrieved_ids: List[str]
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "per_case": []}

        per_case = []
        for item in dataset:
            expected_ids = item.get("expected_retrieval_ids", [])
            retrieved_ids = item.get("retrieved_ids", [])
            metrics = self.evaluate_case(expected_ids, retrieved_ids, top_k=top_k)
            per_case.append(
                {
                    "question": item.get("question", ""),
                    "expected_retrieval_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    **metrics,
                }
            )

        avg_hit_rate = sum(x["hit_rate"] for x in per_case) / len(per_case)
        avg_mrr = sum(x["mrr"] for x in per_case) / len(per_case)
        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "per_case": per_case,
        }
