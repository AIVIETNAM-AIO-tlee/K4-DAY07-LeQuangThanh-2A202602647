# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Quang Thành
**Nhóm:** ColdBrew
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:*
Có nghĩa 2 văn bản có độ tương đồng về ngữ nghĩa (semantic) cao, cosine càng lớn thì semantic càng tương đồng và ngược lại.

**Ví dụ có độ tương tự CAO:**
- Câu A: Chính sách hoàn tiền
- Câu B: Quy định đổi trả
- Tại sao tương đồng: Tương đồng vì ngữ nghĩa của "đổi trả" và "hoàn tiền" đều mang nghĩa và mục đích chung là "trả hàng / trả tiền".

**Ví dụ có độ tương tự THẤP:**
- Câu A: Tôi thích chơi đá bóng
- Câu B: Cây bàng xanh tốt
- Tại sao khác: Vì 2 câu này không có từ nào mang ngữ nghĩa và mục đích giống nhau, dẫn đến khi embed và tính cosine sẽ cho ra result thấp.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:*
Cosine được ưu tiên hơn trong trường hợp semantic word, khi mà yêu cầu độ tương đồng về nghĩa mặc dù 2 câu khác nhau, thiên hướng nội dung. Còn đối với Euclidean distance thì yêu cầu thông tin thật, có cùng 1 đơn vị hoặc chiều, và có khoảng cách, không được embed để cho ra số liệu khác.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* $\frac{10000}{450} = \lceil{22,2223\rceil}$
> *Đáp án:* 23

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:*
Khi độ chồng chéo (overlap) tăng lên 100, số lượng chunk theo đó cũng tăng lên.
Muốn độ chồng chéo nhiều hơn để tăng mức lưu context trong trường hợp AI lấy data từ các chunk để tránh bị thiếu thông tin và bị hallucinate.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*
sentences = re.split(r'[.!?]\s+', text)
Xử lý trường hợp edge case là text rỗng thì trả về chunk rỗng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?*
Thuật toán hoạt động theo recursive gọi split fixed size chunk (`FixedSizeChunker`) tùy theo user, base là 500. Base case để dừng recursive là text rỗng thì return []

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*
- `add_documents`: Mỗi `Document` được chuẩn hóa qua `_make_record`, tạo vector embedding bằng `_embedding_fn` và lưu dưới dạng dictionary vào bộ nhớ trong (`self._store`), đồng thời lưu `doc_id` của tài liệu gốc vào metadata để phục vụ việc xóa hoặc truy vết.
- `search`: Sử dụng helper `_search_records` để embed câu truy vấn, sau đó tính độ tương đồng bằng tích vô hướng (`_dot`) giữa vector query và vector embedding của từng tài liệu đã lưu (do các vector đã được chuẩn hóa đơn vị nên tích vô hướng tương đương cosine similarity), cuối cùng sắp xếp điểm giảm dần để lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*
- `search_with_filter`: Thực hiện lọc metadata **TRƯỚC** khi tìm kiếm (pre-filtering), chỉ giữ lại các bản ghi trong `self._store` thỏa mãn toàn bộ điều kiện trong `metadata_filter`, rồi mới chuyển các bản ghi này vào `_search_records` để tính độ tương đồng. Cách này đảm bảo không làm mất các kết quả phù hợp khi giới hạn top-k.
- `delete_document`: Lọc và loại bỏ khỏi `self._store` những bản ghi có `metadata['doc_id']` hoặc `id` trùng khớp với `doc_id` cần xóa, trả về `True` nếu có ít nhất một bản ghi bị xóa và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*
- Đầu tiên kiểm tra store, nếu rỗng thì trả về thông báo không có thông tin. Nếu có dữ liệu, gọi `store.search(question, top_k)` để truy xuất các đoạn văn bản liên quan nhất.
- Ngữ cảnh được đưa vào prompt bằng cách định dạng từng đoạn có đánh số thứ tự kèm thông tin nguồn: `[i] (Source: ...)\n{content}`, sau đó ghép với câu hỏi kèm chỉ thị yêu cầu mô hình chỉ trả lời dựa vào ngữ cảnh và trích dẫn số thứ tự nguồn (Source Traceability).


---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
```

**Số lượng bài test vượt qua (pass):** __ / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Chính sách đổi trả hàng hóa | Quy định về việc trả hàng và hoàn tiền | cao | 0.8371 | Đúng |
| 2 | Thời gian giao hàng dự kiến từ 2 đến 4 ngày | Đơn hàng sẽ được vận chuyển trong vòng 2-4 ngày làm việc | cao | 0.9043 | Đúng |
| 3 | Người mua có quyền khiếu nại sản phẩm | Người bán có trách nhiệm đóng gói hàng cẩn thận | thấp | 0.7374 | Sai |
| 4 | Shopee hỗ trợ nhiều phương thức thanh toán | Cây bàng trước sân trường đang thay lá | thấp | 0.4931 | Đúng |
| 5 | Hàng hóa bị hư hỏng do vận chuyển | Sản phẩm còn nguyên vẹn và không bị trầy xước | thấp | 0.7135 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*
Kết quả bất ngờ nhất là ở Cặp 5: dù hai câu có ý nghĩa logic hoàn toàn trái ngược nhau ("hư hỏng do vận chuyển" vs "nguyên vẹn không trầy xước"), điểm tương đồng cosine thực tế vẫn đạt rất cao là **0.7135**. Điều này cho thấy mô hình embedding biểu diễn ngữ nghĩa dựa trên **ngữ cảnh và chủ đề xuất hiện** (cùng xoay quanh tình trạng vật lý của kiện hàng trong thương mại điện tử) hơn là phân biệt tính đối lập hoặc phủ định logic. Ngoài ra, ngay cả hai câu hoàn toàn khác biệt ở Cặp 4 vẫn có điểm nền khoảng ~0.49 do hiệu ứng tập trung hình nón (cone effect) đặc trưng trong không gian vector nhúng của mô hình ngôn ngữ.


---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua có thể gửi yêu cầu trả hàng và hoàn tiền trong vòng bao nhiêu ngày kể từ khi nhận hàng? | `77251#26`: Người Mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 ngày kể từ lúc đơn hàng giao thành công (thực phẩm tươi sống là 24h). | 0.8709 | Có | Người Mua được gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 ngày kể từ khi đơn hàng giao thành công. |
| 2 | Điều kiện để Người mua được yêu cầu trả hàng do không còn nhu cầu là gì? | `77251#24`: Sản phẩm ở trạng thái nguyên vẹn và nguyên bao bì nhưng Người Mua không còn nhu cầu ("Trả hàng COM"). | 0.8377 | Có | Sản phẩm phải còn nguyên vẹn, nguyên bao bì/tem mác và thuộc diện chính sách trả hàng COM. |
| 3 | Quy định về việc khiếu nại và đổi trả đối với Người mua (buyer) là gì? | `77251#20`: Các trường hợp khiếu nại đổi trả như Người Bán giao sai sản phẩm, hàng lỗi, hư hỏng, hàng giả hoặc không đúng mô tả. | 0.8002 | Có | Người mua có quyền khiếu nại đổi trả khi nhận sai hàng, hàng lỗi, hàng giả hoặc khác biệt mô tả theo chính sách Shopee. |
| 4 | Người bán có được phép tự tổ chức vận chuyển hàng hóa trên sàn Shopee không? | `77250#1`: Chính sách vận chuyển - quy định trường hợp Người Bán tự tổ chức vận chuyển thì phải tự chịu trách nhiệm trước pháp luật và Người Mua. | 0.8264 | Có | Người Bán có thể tự vận chuyển nhưng phải tự chịu hoàn toàn trách nhiệm pháp lý và bảo đảm quyền lợi Người Mua. |
| 5 | Shopee hỗ trợ giải quyết tranh chấp, khiếu nại giữa Người Mua và Người Bán thông qua cơ chế nào? | `77265#8`: Tranh chấp hoặc khiếu nại được xử lý theo trình tự các bước hỗ trợ của Shopee và cơ chế Thời Gian Shopee Đảm Bảo. | 0.9054 | Có | Shopee tiếp nhận và xử lý tranh chấp theo các bước hòa giải, dựa trên cơ chế Thời Gian Shopee Đảm Bảo. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*
Qua buổi demo và so sánh thực tế giữa các thành viên, tôi học được rằng việc tiền xử lý và lọc bằng siêu dữ liệu (metadata filtering theo `audience: buyer/seller`) là cực kỳ thiết yếu trong các hệ thống chính sách lớn, giúp loại bỏ triệt để việc nhầm lẫn ngữ cảnh giữa hai đối tượng người dùng dù họ dùng chung nhiều từ khóa. Bên cạnh đó, chiến lược `RecursiveChunker` cho thấy khả năng bảo toàn cấu trúc văn bản và ngữ cảnh mạch lạc vượt trội so với việc chia cố định kích thước (`FixedSizeChunker`).

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

