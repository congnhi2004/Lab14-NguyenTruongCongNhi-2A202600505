# 👤 Cá nhân Reflection - Nhị

## 1. Đóng góp kỹ thuật (Engineering Contribution)
Trong dự án này, tôi chịu trách nhiệm chính về mảng **Dữ liệu (Data)** và **Phát triển Agent**:
- **Thiết kế Pipeline SDG:** Xây dựng script `synthetic_gen.py` để tạo ra bộ Golden Dataset gồm 110 test cases chất lượng cao, bao gồm đầy đủ Ground Truth cho cả phần Answer và Metadata của tài liệu.
- **Phát triển Agent V2 (LangGraph):** Triển khai kiến trúc ReAct Agent sử dụng thư viện LangGraph. So với phiên bản V1 (Basic RAG), Agent V2 có khả năng suy nghĩ (Thinking process) và gọi tool tìm kiếm nhiều lần để lọc thông tin chính xác hơn.
- **Quản lý Vector DB:** Đảm bảo quá trình Ingestion tài liệu vào ChromaDB diễn ra suôn sẻ với cấu trúc metadata đầy đủ để hỗ trợ việc lọc (filtering) khi Agent thực thi.

## 2. Chiều sâu kỹ thuật (Technical Depth)
- **LangGraph & ReAct:** Tôi đã áp dụng thành công kỹ thuật ReAct (Reason + Act) giúp Agent giảm thiểu tình trạng Hallucination bằng cách bắt nó phải giải trình lý do trước khi đưa ra câu trả lời.
- **Data Quality:** Tôi hiểu rằng chất lượng của bộ Eval phụ thuộc hoàn toàn vào SDG. Việc thiết kế bộ câu hỏi đa dạng (Easy, Medium, Hard) giúp hệ thống benchmark phản ánh đúng năng lực thực tế của Agent trong các điều kiện khắc nghiệt.

## 3. Giải quyết vấn đề (Problem Solving)
- **Vấn đề:** Trong quá trình chạy V2, Agent đôi khi lặp lại (loop) vô hạn khi không tìm thấy tài liệu.
- **Giải pháp:** Tôi đã tinh chỉnh `SYSTEM_PROMPT` và giới hạn số bước suy nghĩ trong LangGraph để Agent biết dừng lại và thừa nhận khi không có đủ thông tin, thay vì cố gắng suy đoán bừa bãi.
- **Vấn đề Ingestion:** Xử lý việc mất dữ liệu khi restart bằng cách triển khai tính năng `persist_path` cho ChromaDB.
