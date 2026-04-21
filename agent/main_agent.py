import asyncio
from typing import List, Dict

MOCK_VECTOR_DB = {
    "doc_001": "Chính sách nhân sự: Nhân viên chính thức được nghỉ 12 ngày phép mỗi năm. Nghỉ ốm cần giấy chứng nhận bệnh viện tuyến huyện trở lên.",
    "doc_002": "Chính sách bảo mật IT: Mật khẩu tối thiểu 12 ký tự, đổi định kỳ 90 ngày. Cấm sao chép dữ liệu công ty sang USB cá nhân.",
    "doc_003": "Chính sách tài chính công tác: Khách sạn được thanh toán tối đa 1.5 triệu VND mỗi đêm. Không thanh toán các hóa đơn Grab cá nhân cuối tuần.",
    "doc_004": "Chính sách làm việc từ xa: Nhân viên WFH phải dùng Cisco AnyConnect VPN và OTP từ ứng dụng điện thoại."
}

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self):
        self.name = "SupportAgent-v1"

    def _tokenize(self, text: str) -> set:
        return {token.strip(".,:;!?()[]\"'").lower() for token in text.split() if token.strip()}

    def simple_retrieve(self, question: str, top_k: int = 1) -> List[str]:
        q_tokens = self._tokenize(question)
        scores = []
        for doc_id, chunk in MOCK_VECTOR_DB.items():
            c_tokens = self._tokenize(chunk)
            overlap = len(q_tokens.intersection(c_tokens))
            scores.append((doc_id, overlap))
        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores or scores[0][1] == 0:
            return []
        return [doc_id for doc_id, _ in scores[:top_k]]

    def _build_answer(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "Tôi không tìm thấy thông tin này trong tài liệu hướng dẫn nội bộ."
        if "mật khẩu" in question.lower():
            return "Mật khẩu phải có ít nhất 12 ký tự và cần đổi sau mỗi 90 ngày."
        if "nghỉ" in question.lower():
            return "Nhân viên chính thức được nghỉ 12 ngày phép/năm; nghỉ ốm cần giấy chứng nhận phù hợp."
        return contexts[0]

    async def query(self, question: str) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Tìm kiếm context liên quan.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """
        await asyncio.sleep(0.2)
        retrieved_ids = self.simple_retrieve(question, top_k=1)
        retrieved_contexts = [MOCK_VECTOR_DB[doc_id] for doc_id in retrieved_ids]
        answer = self._build_answer(question, retrieved_contexts)

        return {
            "answer": answer,
            "contexts": retrieved_contexts,
            "metadata": {
                "model": "support-agent-v1",
                "tokens_used": max(50, len(question.split()) * 5 + 40),
                "sources": retrieved_ids
            }
        }


class MainAgentV2(MainAgent):
    def __init__(self):
        super().__init__()
        self.name = "SupportAgent-v2"

    def _expand_query(self, question: str) -> str:
        synonyms = {
            "đổi mật khẩu": "mật khẩu bảo mật 90 ngày 12 ký tự",
            "làm việc từ xa": "WFH VPN OTP Cisco AnyConnect",
            "công tác": "khách sạn thanh toán 1.5 triệu",
        }
        expanded = question
        lower_q = question.lower()
        for key, value in synonyms.items():
            if key in lower_q:
                expanded = f"{question} {value}"
        return expanded

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(0.25)
        expanded_question = self._expand_query(question)
        retrieved_ids = self.simple_retrieve(expanded_question, top_k=3)
        retrieved_contexts = [MOCK_VECTOR_DB[doc_id] for doc_id in retrieved_ids]
        answer = self._build_answer(question, retrieved_contexts)
        return {
            "answer": answer,
            "contexts": retrieved_contexts,
            "metadata": {
                "model": "support-agent-v2",
                "tokens_used": max(60, len(expanded_question.split()) * 5 + 45),
                "sources": retrieved_ids
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
