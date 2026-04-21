# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 56
- **Tỉ lệ Pass/Fail:** 36/20 (Pass rate: 64.29%)
- **Retrieval metrics trung bình:**
    - Hit Rate: 0.8214
    - MRR: 0.8214
- **Agreement Rate (Multi-Judge):** 0.75
- **Điểm LLM-Judge trung bình:** 3.0357 / 5.0
- **Hiệu năng & chi phí:**
    - Avg latency: 0.251s/case
    - Cost estimate: 0.000227 USD/case

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Retrieval Miss | 10 | Query expansion chưa đủ tốt cho câu hỏi diễn đạt khác mẫu |
| Refusal Handling | 6 | Một số case adversarial chưa từ chối ổn định ở cả 2 judge |
| Partial Answer | 4 | Câu trả lời thiếu 1 vế trong case multi-hop/cross-document |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Retrieval miss cho câu hỏi diễn đạt khác template
1. **Symptom:** Agent không lấy đúng doc cho câu hỏi cùng nghĩa nhưng khác từ vựng.
2. **Why 1:** Retriever dựa lexical overlap đơn giản.
3. **Why 2:** Không có semantic reranker phía sau top-k retrieval.
4. **Why 3:** Query expansion mới hard-code vài mẫu.
5. **Why 4:** Chưa có vòng phản hồi từ failure set để cập nhật synonym map.
6. **Root Cause:** Recall của Retrieval chưa đủ robust với paraphrase.

### Case #2: Trả lời thiếu trong case multi-hop
1. **Symptom:** Agent chỉ trả một phần quy định khi câu hỏi yêu cầu tổng hợp 2 chính sách.
2. **Why 1:** Generation ưu tiên context đầu tiên.
3. **Why 2:** Không có prompt bắt buộc "cover all retrieved evidence".
4. **Why 3:** Chưa có hậu kiểm completeness trước khi trả lời.
5. **Why 4:** Judge chấm điểm vẫn pass nếu đúng một phần.
6. **Root Cause:** Prompting và post-check chưa tối ưu cho multi-hop synthesis.

### Case #3: Refusal chưa nhất quán với prompt tấn công
1. **Symptom:** Một số câu hỏi adversarial không bị từ chối đủ mạnh.
2. **Why 1:** Bộ từ khóa phát hiện unsafe/ooc còn hạn chế.
3. **Why 2:** Không dùng classifier intent riêng trước bước answer generation.
4. **Why 3:** Chính sách từ chối chưa được mã hóa thành rule cứng.
5. **Why 4:** Thiếu bộ test red-team rộng để khóa hành vi.
6. **Root Cause:** Safety guardrail hiện tại dựa heuristic mỏng.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Tăng recall retrieval bằng hybrid search (BM25 + embedding) và thêm reranker.
- [ ] Thêm "answer completeness checklist" cho case multi-hop trước khi trả kết quả.
- [ ] Bổ sung classifier cho OOD/adversarial để ép nhánh refusal sớm.
- [ ] Mở rộng synthetic hard-cases mỗi sprint và theo dõi delta theo nhóm lỗi.
- [ ] Mục tiêu vòng tới: +5 điểm hit_rate, giữ latency < 0.30s/case, giảm 30% cost bằng cache judge và batch inference.
