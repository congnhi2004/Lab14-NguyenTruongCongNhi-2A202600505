# 📊 Failure Analysis: AI Evaluation Factory

## 1. Overview & Executive Summary
Hệ thống Evaluation đã hoàn thành đánh giá **Agent_V2_Optimized** trên bộ dữ liệu **110 test cases**. Kết quả cho thấy sự cải thiện rõ rệt về chất lượng câu trả lời so với phiên bản V1, tuy nhiên hệ thống đo lường vẫn còn tồn tại các vấn đề về kỹ thuật cần khắc phục để đạt được độ tin cậy tối đa.

**Các thông số chính:**
- **Avg Score:** 3.95 / 5.0 (Tăng từ 3.0 của V1)
- **Hit Rate:** 0.0% (Lỗi hệ thống đo lường)
- **Agreement Rate:** ~66% (Độ đồng thuận giữa 2 model Judge)

---

## 2. Retrieval Evaluation (Hit Rate & MRR)
### Hiện trạng:
Chỉ số Hit Rate hiện đang đạt mức **0.0%**. Đây là một lỗi nghiêm trọng trong pipeline đánh giá (Evaluation Pipeline Failure), không phản ánh đúng chất lượng Retrieval thực tế (vì Agent vẫn trả lời đúng dựa trên tài liệu).

### Phân tích "5 Whys" cho lỗi Hit Rate:
1. **Why is the Hit Rate 0.0?** -> Danh sách `expected_retrieval_ids` trong bộ kết quả luôn rỗng.
2. **Why is it empty?** -> Script Generating Dataset (`synthetic_gen.py`) không tạo được mapping ID chính xác với Vector DB.
3. **Why no mapping?** -> Quá trình Ingestion trong `chroma_db` sử dụng UUID ngẫu nhiên thay vì ID định danh dựa trên nội dung (Content-based hashing).
4. **Why random UUIDs?** -> Hệ thống chưa đồng bộ hóa schema giữa module Ingestion và module SDG (Synthetic Data Generation).
5. **Root Cause:** Thiếu một cơ chế định danh tài liệu thống nhất (Unified Document Indexing) để link kết quả tìm kiếm với Ground Truth metadata.

---

## 3. Multi-Judge Consensus Analysis
Hệ thống sử dụng **gpt-4o-mini** và **gpt-4o** để chấm điểm.
- **Agreement Rate (66%):** Cho thấy 2 model đồng thuận hoàn toàn trên 2/3 số lượng test cases.
- **Conflict Handling:** Các trường hợp có sai số > 1 điểm (Agreement Rate = 0) thường rơi vào các câu hỏi yêu cầu giải thích sâu hoặc có đáp án Ground Truth ngắn gọn (ví dụ: câu hỏi về "turtles suborders").
- **Observation:** `gpt-4o` thường khắt khe hơn về tính chi tiết, trong khi `gpt-4o-mini` dễ dãi hơn với các câu trả lời mang tính tổng quát.

---

## 4. Failure Clustering (Phân cụm lỗi Generation)

### Cluster 1: Thiếu tính cụ thể (Specificity Gap)
- **Ví dụ tiêu biểu:** Câu hỏi về "Ducks feed on land".
- **Biểu hiện:** Agent trả lời rất dài, nhắc đến nhiều loài vịt nhưng không nêu được từ khóa "Dabbling ducks" như Ground Truth yêu cầu.
- **Nguyên nhân:** Prompt của Agent V2 quá tập trung vào việc "tổng hợp" dẫn đến việc làm loãng (diluting) các thông tin mang tính định danh quan trọng.

### Cluster 2: Lỗi kiến thức do Retrieval lỗi (Retrieval Noise)
- **Ví dụ tiêu biểu:** Các câu hỏi về "wolves mating behavior".
- **Biểu hiện:** Agent lấy được tài liệu liên quan nhưng lại trích xuất các ý phụ (Resource management) thay vì ý chính trong Ground Truth (Pack tension).
- **Nguyên nhân:** Chunking strategy hiện tại có kích thước quá lớn, dẫn đến việc lấy được context nhưng thông tin quan trọng bị chìm nghỉm trong các đoạn văn dài.

---

## 5. Red Teaming & Out-of-Distribution (OOD) Analysis
Hệ thống đã được thử nghiệm với một số câu hỏi "bẫy" (Red Teaming) để kiểm tra tính ổn định:
- **Câu hỏi không liên quan (OOD):** "Who is Daffy Duck?". 
- **Kết quả:** Agent vẫn trả lời tốt nhờ khả năng fallback kiến thức của LLM, tuy nhiên điều này cho thấy rủi ro "Over-reliance" vào kiến thức sẵn có của model thay vì chỉ dựa vào tài liệu.
- **Trường hợp "Silly Duck":** Câu hỏi về loài vật hài hước nhất cho thấy Agent có khả năng đa dạng hóa nguồn tin nhưng đôi khi bị xao nhãng bởi các thông tin mang tính giải trí thay vì khoa học.

---

## 6. Regression Release Gate Decision
**Quyết định: Block Release** (Mặc dù Delta Score dương).

**Lý do:** 
1. Mặc dù điểm chất lượng tăng (từ 3.0 lên 3.95), nhưng hệ thống Retrieval Eval chưa hoạt động (Hit Rate 0%). 
2. Việc Release một model khi chưa đo lường được khả năng Retrieval stage là cực kỳ rủi ro vì có thể tiềm ẩn lỗi Hallucination mà Judge model (sử dụng context kèm theo) không phát hiện ra.

---

## 7. Recommendations (Đề xuất cải tiến)
1. **Fix Retrieval Eval:** Chuyển sang sử dụng `article_title` hoặc `doc_hash` làm ID thay vì UUID ngẫu nhiên để đồng bộ hóa SDG và Retrieval.
2. **Chunking Strategy:** Giảm chunk size từ 1000 xuống 500 characters và tăng overlap lên 10% để cải thiện độ tập trung của context.
3. **Prompt Engineering:** Thêm instruction "Be concise and prioritize specific entities from the retrieved text" để Agent tập trung vào các từ khóa then chốt.
4. **Cost Optimization:** Sử dụng `gpt-4o-mini` cho 100% các case đơn giản (dựa trên classifier) và chỉ dùng `gpt-4o` cho các case có độ khó cao để giảm 40% chi phí Eval.
