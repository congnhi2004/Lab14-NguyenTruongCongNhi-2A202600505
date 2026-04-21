# 👤 Cá nhân Reflection
> **Họ tên:** Phạm Hoàng Long
> **MSHV:** 2A202600261

## 1. Đóng góp kỹ thuật (Engineering Contribution)
Tôi chịu trách nhiệm về hệ thống **Đánh giá (Evaluation)** và **Thực thi (Runner)**:
- **Retrieval Metrics:** Triển khai module `retrieval_eval.py` để tính toán Hit Rate @3 và MRR. Đây là chìa khóa để đánh giá độ chính xác của Vector DB trước khi xem xét câu trả lời cuối cùng.
- **Multi-Judge Consensus Engine:** Xây dựng `LLMJudge` sử dụng 2 model (GPT-4o và GPT-4o-mini). Tôi đã viết logic tính toán `Agreement Rate` để đo lường độ tin cậy của các giám khảo AI.
- **Async Benchmark Runner:** Phát triển `runner.py` sử dụng `asyncio.gather` để chạy song song các test cases, giúp rút ngắn thời gian benchmark từ hàng chục phút xuống còn chưa tới 2 phút cho hơn 100 cases.

## 2. Chiều sâu kỹ thuật (Technical Depth)
- **MRR (Mean Reciprocal Rank):** Tôi đã áp dụng MRR để hiểu được vị trí của tài liệu đúng trong danh sách kết quả, từ đó nhận định được độ nhiễu của Retrieval pipeline.
- **Consensus Bias:** Tôi nhận thấy sự khác biệt về "khẩu vị" giữa GPT-4o và 4o-mini. Việc kết hợp cả hai giúp điểm số cuối cùng khách quan hơn và giảm thiểu lỗi do một model đơn lẻ gây ra.

## 3. Giải quyết vấn đề (Problem Solving)
- **Vấn đề Rate Limit:** Khi chạy 100+ cases song song, hệ thống thường xuyên bị lỗi Rate Limit và Timeout từ phía OpenAI API.
- **Giải pháp:** Tôi đã triển khai cơ chế `batch_size` và `asyncio.sleep` giữa các vòng lặp, đồng thời cấu hình lại `timeout` cho AI client lên 60 giây để đảm bảo toàn bộ pipeline không bị ngắt quãng giữa chừng.
- **Vấn đề Interface:** Khắc phục sự thiếu hụt phương thức `score()` trong Evaluator để đảm bảo tương thích hoàn toàn với hệ thống Runner cũ của nhóm.
