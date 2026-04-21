import asyncio
import json
import os
import time
from typing import Dict, List, Tuple
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent, MainAgentV2

class ExpertEvaluator:
    def __init__(self, top_k: int = 3):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.top_k = top_k

    @staticmethod
    def _token_overlap_ratio(a: str, b: str) -> float:
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not b_tokens:
            return 0.0
        return len(a_tokens.intersection(b_tokens)) / len(b_tokens)

    async def score(self, case, resp):
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = resp.get("metadata", {}).get("sources", [])
        retrieval = self.retrieval_evaluator.evaluate_case(expected_ids, retrieved_ids, top_k=self.top_k)

        answer = resp.get("answer", "")
        expected_answer = case.get("expected_answer", "")
        faithfulness = min(1.0, self._token_overlap_ratio(answer, expected_answer) + 0.1)
        relevancy = min(1.0, self._token_overlap_ratio(answer, case.get("question", "")) + 0.2)

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": retrieval,
        }


def load_dataset(path: str = "data/golden_set.jsonl") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thiếu {path}. Hãy chạy 'python data/synthetic_gen.py' trước.")

    with open(path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        raise ValueError(f"File {path} rỗng. Hãy tạo ít nhất 1 test case.")
    return dataset


def build_summary(results: List[Dict], agent_version: str, elapsed_seconds: float) -> Dict:
    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    agreement_rate = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total
    avg_tokens = sum(r.get("tokens_used", 0) for r in results) / total
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    estimated_cost_usd = total_tokens * 0.000002
    cost_per_case_usd = estimated_cost_usd / total
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_seconds, 4),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(pass_rate, 4),
            "hit_rate": round(hit_rate, 4),
            "mrr": round(avg_mrr, 4),
            "agreement_rate": round(agreement_rate, 4),
            "avg_latency_seconds": round(avg_latency, 4),
            "avg_tokens": round(avg_tokens, 2),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "cost_per_case_usd": round(cost_per_case_usd, 6),
        },
    }


def should_release(v1_summary: Dict, v2_summary: Dict) -> Tuple[bool, Dict]:
    v1m = v1_summary["metrics"]
    v2m = v2_summary["metrics"]
    checks = {
        "quality_score_non_decrease": v2m["avg_score"] >= v1m["avg_score"] - 0.05,
        "retrieval_hit_rate_non_decrease": v2m["hit_rate"] >= v1m["hit_rate"] - 0.02,
        "agreement_minimum": v2m["agreement_rate"] >= 0.65,
        "latency_regression_guard": v2m["avg_latency_seconds"] <= v1m["avg_latency_seconds"] * 1.25,
        "cost_regression_guard": v2m["cost_per_case_usd"] <= v1m["cost_per_case_usd"] * 1.25,
    }
    return all(checks.values()), checks


async def run_benchmark_with_results(agent, agent_version: str, dataset: List[Dict]):
    print(f"🚀 Khởi động Benchmark cho {agent_version} với {len(dataset)} cases...")
    start_time = time.perf_counter()
    runner = BenchmarkRunner(agent, ExpertEvaluator(top_k=3), LLMJudge())
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - start_time
    summary = build_summary(results, agent_version=agent_version, elapsed_seconds=elapsed)
    return results, summary


async def main():
    try:
        dataset = load_dataset("data/golden_set.jsonl")
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        return

    v1_results, v1_summary = await run_benchmark_with_results(MainAgent(), "Agent_V1_Base", dataset)
    v2_results, v2_summary = await run_benchmark_with_results(MainAgentV2(), "Agent_V2_Optimized", dataset)

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    hit_delta = v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"]
    cost_delta = v2_summary["metrics"]["cost_per_case_usd"] - v1_summary["metrics"]["cost_per_case_usd"]
    latency_delta = v2_summary["metrics"]["avg_latency_seconds"] - v1_summary["metrics"]["avg_latency_seconds"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.4f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.4f}")
    print(f"Delta Score: {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"Delta HitRate: {'+' if hit_delta >= 0 else ''}{hit_delta:.4f}")
    print(f"Delta Cost/Case: {'+' if cost_delta >= 0 else ''}{cost_delta:.6f} USD")
    print(f"Delta Latency: {'+' if latency_delta >= 0 else ''}{latency_delta:.4f} s")

    release, release_checks = should_release(v1_summary, v2_summary)

    os.makedirs("reports", exist_ok=True)
    full_summary = {
        "metadata": v2_summary["metadata"],
        "metrics": v2_summary["metrics"],
        "regression": {
            "baseline_version": v1_summary["metadata"]["version"],
            "candidate_version": v2_summary["metadata"]["version"],
            "delta": {
                "avg_score": round(delta, 4),
                "hit_rate": round(hit_delta, 4),
                "cost_per_case_usd": round(cost_delta, 6),
                "avg_latency_seconds": round(latency_delta, 4),
            },
            "release_decision": "APPROVE" if release else "ROLLBACK",
            "release_checks": release_checks,
        },
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": {"summary": v1_summary, "results": v1_results},
                "candidate": {"summary": v2_summary, "results": v2_results},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if release:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (ROLLBACK)")

if __name__ == "__main__":
    asyncio.run(main())
