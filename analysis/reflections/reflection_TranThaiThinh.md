# Reflection cá nhân - Trần Thái Thịnh - 2A202600310 - LÀM CÁ NHÂN DAY 14

## 1) Vai trò và phần việc chính
- Vai trò trong nhóm: phụ trách chính end-to-end toàn bộ pipeline Evaluation Factory.
- Module phụ trách: `data/synthetic_gen.py`, `engine/retrieval_eval.py`, `engine/llm_judge.py`, `engine/runner.py`, `main.py`, `analysis/failure_analysis.md`, và bộ `reports/*`.
- Commit/PR tiêu biểu: hoàn thiện SDG 56 cases, triển khai Retrieval metrics (Hit Rate/MRR), thêm consensus 2-judge + conflict handling, thêm regression release gate theo chất lượng/chi phí/hiệu năng.

## 2) Đóng góp kỹ thuật cụ thể
- Retrieval/SDG: xây dựng generator tạo >50 cases có `expected_retrieval_ids`, bao gồm cả single-hop, multi-hop, adversarial, out-of-context để đo đúng retrieval trước generation.
- Multi-Judge/Calibration: triển khai cơ chế chấm điểm từ 2 judge mô phỏng (GPT/Claude), tính `agreement_rate`, phát hiện chênh lệch và áp dụng `conflict_penalty` khi điểm lệch lớn.
- Regression Gate/DevOps: chạy song song benchmark cho V1/V2, tính delta metrics (score/hit rate/cost/latency), và quyết định `APPROVE` hoặc `ROLLBACK` theo ngưỡng tự động.

## 3) Vấn đề gặp phải và cách xử lý
- Vấn đề 1: mã nguồn ban đầu mới là placeholder, chưa có logic đo retrieval thực tế và chưa có gate release.
- Cách xử lý: thiết kế lại luồng đánh giá theo từng lớp (agent -> retrieval eval -> multi-judge -> summary/regression), chuẩn hóa output schema để tổng hợp metric nhất quán.
- Kết quả: pipeline chạy ổn định với dataset 56 cases; xuất đầy đủ `reports/summary.json`, `reports/benchmark_results.json`; `python check_lab.py` pass.

## 4) Điều học được
- Kiến thức kỹ thuật: hiểu rõ tách bạch evaluation retrieval và generation, cách đo Hit Rate/MRR, ý nghĩa agreement trong multi-judge, và cách thiết kế release gate bằng threshold.
- Bài học phối hợp nhóm: khi một người làm full-stack dự án, cần khóa schema dữ liệu từ sớm và viết báo cáo theo số liệu thật để giảm rework ở giai đoạn nộp.

## 5) Kế hoạch cải thiện cá nhân
- Kỹ năng cần nâng cấp: tích hợp judge thật qua API (thay mô phỏng heuristic), tối ưu thêm chi phí eval mà vẫn giữ độ ổn định.
- Hành động cụ thể tuần tới: bổ sung reranker semantic cho retrieval, thêm test tự động cho từng module metric, và đặt mục tiêu giảm thêm 30% cost/case bằng batch + cache.
