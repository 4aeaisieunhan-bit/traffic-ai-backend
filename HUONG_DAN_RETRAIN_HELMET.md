# Hướng dẫn: lấy dữ liệu từ web → kiểm tra nhãn → retrain mô hình mũ bảo hiểm

Mục tiêu: sửa lỗi AI nhầm **mũ đen** (hoặc vật màu tối trên đầu) thành **không đội mũ**, bằng cách dùng chính kết quả bạn duyệt vi phạm trên web làm dữ liệu dạy lại mô hình `weights/helmet_26s.pt`.

> Mọi lệnh trong file này chạy trong thư mục backend, đã bật môi trường ảo:
> ```powershell
> cd D:\Project\traffic-ai-backend
> .venv\Scripts\activate
> $env:PYTHONIOENCODING = "utf-8"     # tránh lỗi UnicodeEncodeError khi in emoji / tiếng Việt
> ```

---

## 0. Toàn bộ quy trình

```
 Phân tích video trên web
        │  (AI phát hiện "Không đội mũ" -> tự lưu mẫu thô: static/evidence/train_*.jpg + .json)
        ▼
 Duyệt vi phạm trên web ──► Xác nhận            -> nhãn no_helmet  ┐
                       └──► Có Đội Mũ (AI Nhầm) -> nhãn helmet     ├─► dataset/helmet_feedback/
                                                                   ┘
 (tuỳ chọn) tools/helmet_mine.py -> sửa nhãn -> dataset/helmet_reviewed/
        │
        ▼
 python tools/helmet_train.py   (gộp dữ liệu -> train tiếp -> tự so sánh cũ / mới)
        │
        ▼
 Kiểm tra trên video MỚI -> nếu tốt: đặt HELMET_WEIGHTS trong .env -> khởi động lại backend
```

**Bấm nút trên web KHÔNG tự retrain.** Nó chỉ tạo dữ liệu đã gán nhãn. Mô hình chỉ thay đổi khi bạn tự chạy script train ở mục 5, và chỉ được web dùng khi bạn tự đổi `.env` ở mục 8.

---

## 1. Chạy hệ thống

| | Lệnh | Kiểm tra |
|---|---|---|
| Backend | `uvicorn server:app --port 8000` | http://localhost:8000/docs |
| Frontend | `cd D:\Project\traffic-ai-frontend` rồi `npm start` | http://localhost:4200 |

Backend khởi động đúng khi thấy dòng `✅ Đã kết nối thành công tới Microsoft SQL Server ...` và `Uvicorn running on http://127.0.0.1:8000`.

---

## 2. Lấy dữ liệu từ web

### 2.1 Phân tích video

Vào **Phân tích video**, upload video, chọn địa điểm, bấm **Bắt đầu phân tích AI**.

Mỗi khi AI kết luận một xe **không đội mũ**, backend lưu thêm một **mẫu thô** vào `static/evidence/`. Code nằm ở `api/analyze.py` (hàm `_save_violation`) và `utils/helmet_dataset.py` (hàm `save_sample`).

| File | Nội dung |
|---|---|
| `train_<task>_<track>_helmet.jpg` | Khung hình 1280x720 đã tăng nét, **chưa vẽ khung**. Đây đúng là ảnh mô hình nhìn thấy khi chạy. |
| `train_<task>_<track>_helmet.json` | Mọi khung đầu mô hình phát hiện ở frame đó (nhãn + độ tin cậy) và khung đầu của người bị kết luận vi phạm. |

Chỉ vi phạm phát hiện **sau khi cập nhật code này** mới có mẫu thô. Vi phạm cũ không dùng làm dữ liệu được.

### 2.2 Duyệt vi phạm

Vào **Vi phạm**, mở một hồ sơ loại **"Không đội mũ bảo hiểm"**, xem ảnh cận cảnh rồi chọn:

| Nút | Khi nào bấm | Nhãn ghi cho người vi phạm |
|---|---|---|
| **Xác Nhận Vi Phạm & Gửi Xử Lý** | Đúng là không đội mũ | `no_helmet` (lớp `1`) |
| **Có Đội Mũ (AI Nhầm)** (nút tím) | Thực ra có đội mũ, ví dụ mũ đen | `helmet` (lớp `0`), loại dữ liệu sửa lỗi quan trọng nhất |
| **Từ Chối Vi Phạm (Hủy Bỏ)** | Sai vì lý do khác (sai xe, sai biển số…) | Không ghi; nếu đã ghi trước đó thì **gỡ ra** |

Nếu bấm nhầm thì cứ bấm lại nút đúng: dữ liệu được ghi đè theo lần duyệt cuối cùng. Code xử lý nằm ở `api/violations.py` (`update_violation_status`) và `utils/helmet_dataset.py` (`apply_review`).

**Mẹo để có dữ liệu tốt:**
- Chỉ bấm "Có Đội Mũ" khi **nhìn rõ** là mũ. Không chắc thì bấm "Từ Chối" thường, vì nhãn sai còn hại hơn thiếu nhãn.
- Nên duyệt **cả** ca đúng (Xác nhận). Nếu chỉ dạy toàn ca "là mũ", mô hình dễ bỏ sót người không đội mũ thật.

---

## 3. Kiểm tra nhãn sau khi bấm

### 3.1 Trên web

Trong trang chi tiết vi phạm, dưới ảnh cận cảnh có dòng trạng thái:

| Dòng hiển thị | Ý nghĩa |
|---|---|
| *Kết quả duyệt hồ sơ này sẽ được dùng để huấn luyện lại…* | Có mẫu thô nhưng chưa duyệt |
| *Đã lưu vào dữ liệu huấn luyện: **CÓ đội mũ** (AI nhận nhầm)* | Đã ghi nhãn `helmet` |
| *Đã lưu vào dữ liệu huấn luyện: **KHÔNG đội mũ** (AI đúng)* | Đã ghi nhãn `no_helmet` |
| (không có dòng nào) | Vi phạm cũ hoặc không phải lỗi mũ, không có mẫu |

Lịch sử duyệt cũng ghi chú: *"Có đội mũ — mô hình nhận nhầm (đã thêm vào dữ liệu huấn luyện)"*.

### 3.2 Trong thư mục `dataset/helmet_feedback/`

```
dataset/helmet_feedback/
├── images/<task>_<track>.jpg     ảnh dùng để train
├── labels/<task>_<track>.txt     nhãn YOLO của MỌI cái đầu trong ảnh
├── meta/<task>_<track>.json      mã VP, trạng thái duyệt, nhãn người vi phạm (để tra cứu)
└── classes.txt                   helmet / no_helmet
```

Mở `labels/<task>_<track>.txt` bằng Notepad. Mỗi dòng là một cái đầu, gồm `lớp  tâm_x  tâm_y  rộng  cao` (tỉ lệ 0–1 so với ảnh):

```
0 0.780469 0.611111 0.010937 0.022222    <- DÒNG ĐẦU = người vi phạm, nhãn do bạn chọn
0 0.152344 0.401389 0.012500 0.025000    <- các đầu khác: nhãn sẵn của AI (độ tin cậy >= 0.45)
1 0.413281 0.522222 0.011719 0.023611
```

- **Dòng đầu tiên** bắt đầu bằng `0` nghĩa là bạn đã bấm **Có Đội Mũ**; bắt đầu bằng `1` nghĩa là bạn đã bấm **Xác nhận**.
- **Ô xám** trong `images/*.jpg` là các đầu mà AI không chắc (độ tin cậy 0.25–0.45). Chúng bị tô xám để không bị coi là "nền" khi train; đây là cố ý, không phải lỗi.
- **Nhanh hơn:** mở `meta/<task>_<track>.json` và xem trường `"violator_label"`.

Xoá vi phạm trên web thì mẫu thô trong `static/evidence/` bị xoá, nhưng **mẫu đã duyệt trong `dataset/` được giữ lại**.

---

## 4. (Nên làm) Gom thêm dữ liệu từ video

Mỗi video chỉ có vài vi phạm mũ, nên chỉ duyệt trên web thì dữ liệu tăng rất chậm. Script sau quét video và giữ các khung hình có đầu **đáng ngờ** (bị báo no_helmet, hoặc AI phân vân), kèm nhãn sẵn để bạn chỉ cần **sửa**:

```powershell
python tools/helmet_mine.py "video 18s.mp4" 21s_vidnew.mov
python tools/helmet_mine.py D:\videos\*.mp4 --every 0.5 --max-images 400
```

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--every` | 0.5 | Lấy 1 khung hình mỗi N giây |
| `--min-head` | 12 | Bỏ đầu nhỏ hơn N px (quá nhỏ để gán nhãn chắc chắn) |
| `--max-images` | 0 | Giới hạn số ảnh (0 = không giới hạn) |
| `--out` | `dataset/helmet_mining` | Thư mục kết quả |

Kết quả nằm trong `dataset/helmet_mining/`:
- **`images/` và `labels/`:** ảnh kèm nhãn sẵn, **chưa được dùng để train**.
- **`review/`:** ảnh phóng to từng đầu đáng ngờ. Tên file có nhãn và độ tin cậy, ví dụ `..._no_helmet_0.76.jpg`, để lướt xem nhanh.

**Sửa nhãn bằng X-AnyLabeling** (miễn phí, đọc được định dạng YOLO):
1. Mở thư mục `dataset/helmet_mining/images`, rồi nạp nhãn YOLO kèm `classes.txt`.
2. Sửa các đầu sai, chủ yếu là mũ đen đang bị gán `no_helmet` thì đổi thành `helmet`. Thêm khung cho đầu bị bỏ sót.
3. Xuất hoặc lưu nhãn YOLO, rồi chép **ảnh đã sửa xong** cùng nhãn vào:
   ```
   dataset/helmet_reviewed/images/*.jpg
   dataset/helmet_reviewed/labels/*.txt     (cùng tên với ảnh)
   ```

Script train **chỉ đọc `helmet_reviewed/`**, không bao giờ dùng nhãn chưa sửa trong `helmet_mining/`.

---

## 5. Retrain

### 5.1 Khi nào nên train

Nên có ít nhất khoảng **200–300 ảnh có mũ đen đã gán đúng**, cộng thêm ca không mũ thật. Ít hơn thì model mới thường chưa tốt hơn rõ, và bước so sánh sẽ báo "CHƯA nên thay".

Đếm nhanh số ảnh đã có:

```powershell
(Get-ChildItem dataset\helmet_feedback\labels\*.txt).Count
(Get-ChildItem dataset\helmet_reviewed\labels\*.txt).Count
```

### 5.2 Lệnh train

```powershell
# Khuyên dùng: trộn thêm bộ dữ liệu gốc để mô hình không "quên" những gì đã học
python tools/helmet_train.py --base-data D:\datasets\Mu_Bao_Hiem_QA-1\data.yaml

# Không có bộ dữ liệu gốc: chỉ dùng dữ liệu từ web + dữ liệu đã sửa
python tools/helmet_train.py
```

`helmet_26s.pt` được train trên bộ **"Mũ_Bảo_Hiểm_QA-1"** (Roboflow, chạy trên Kaggle). Nếu còn bộ này, tải về và dùng `--base-data`. Bộ gốc phải có đúng 2 lớp theo thứ tự `helmet, no_helmet`, nếu không script sẽ báo lỗi.

Script tự làm:

1. **Gộp dữ liệu** vào `dataset/helmet_merged/`. Các nguồn: `helmet_feedback` và `helmet_reviewed`, cộng bộ gốc nếu có `--base-data`.
2. **Chia train / val / test** theo nhóm: cùng một lần phân tích, hoặc cùng đoạn khoảng 3 giây của một video, nằm chung một phần. Nhờ vậy bộ test không chứa frame gần giống frame đã học. Cách chia cố định giữa các lần chạy, nên so sánh giữa các lần train là công bằng.
3. **Train tiếp từ `weights/helmet_26s.pt`:** learning rate nhỏ, giữ nguyên 10 lớp đầu, tăng thay đổi độ sáng để học mũ đen và tóc đen trong nhiều điều kiện ánh sáng.
4. **Lưu model mới** ở `runs/detect/helmet_ft/weights/best.pt`. Lần chạy sau sẽ thành `helmet_ft2`, `helmet_ft3`…
5. **Tự chạy so sánh cũ / mới** (mục 6).

Các tuỳ chọn hay dùng:

| Tuỳ chọn | Mặc định | Khi nào đổi |
|---|---|---|
| `--epochs` | 40 | Dữ liệu nhiều có thể tăng lên 60–80 |
| `--batch` | 8 | Báo `CUDA out of memory` thì giảm xuống 4 |
| `--freeze` | 10 | Dữ liệu rất ít thì tăng lên (giữ nguyên nhiều lớp hơn) |
| `--name` | `helmet_ft` | Đặt tên dễ nhớ, ví dụ `helmet_v2` |
| `--no-eval` | tắt | Bỏ bước so sánh tự động |

Thời gian ước tính trên RTX 4050 với vài trăm ảnh, 40 epoch: khoảng 10–30 phút.

---

## 6. Đọc kết quả so sánh

Cuối quá trình train, script in bảng so sánh (đây là ví dụ minh hoạ, không phải kết quả thật):

```
Bộ test: 85 ảnh
Mô hình                         mAP50  mAP50-95  Mũ->KHÔNG MŨ  Bắt đúng ko mũ  Báo ko mũ thừa
weights\helmet_26s.pt           0.902     0.655         11.8%           84.0%              21
runs\detect\helmet_v2\...\best  0.915     0.668          4.1%           83.0%               9
(Bộ test có 1460 đầu đội mũ, 210 đầu không mũ)
✅ NÊN dùng mô hình mới: ít phạt nhầm hơn, không bỏ sót thêm.
```

| Cột | Ý nghĩa | Model mới cần |
|---|---|---|
| **Mũ->KHÔNG MŨ** | % đầu **có mũ** bị báo không mũ, tức phạt nhầm (gồm lỗi mũ đen) | **Giảm**. Đây là chỉ số chính. |
| **Bắt đúng ko mũ** | % người **không mũ** bị bắt đúng | Không giảm quá 2% |
| **Báo ko mũ thừa** | Báo no_helmet ở chỗ không có đầu nào | Không tăng |
| mAP50 / mAP50-95 | Độ chính xác tổng quát | Tham khảo thêm |

Các chỉ số được đo ở **đúng ngưỡng hệ thống đang dùng** (no_helmet ≥ 0.55, helmet ≥ 0.35), trên ảnh mô hình **chưa từng học**.

Chạy lại phần so sánh bất kỳ lúc nào, không cần train:

```powershell
python tools/helmet_eval.py --weights weights/helmet_26s.pt runs/detect/helmet_v2/weights/best.pt
```

Có thể đưa nhiều model vào cùng lúc; đặt model **cũ trước**, **mới sau**.

Nếu thấy cảnh báo *"Bộ test còn rất ít ảnh"* hoặc *"tạm đánh giá trên tập val"*, kết quả chưa đáng tin. Hãy gán nhãn thêm rồi train lại.

---

## 7. Kiểm tra bằng mắt trên video

Dùng **video mới, chưa từng đưa vào dữ liệu train**. Nếu thử lại đúng video đã gán nhãn, model mới gần như chắc chắn "sửa được" vì đã thấy đúng cái đầu đó khi học; kết quả đó không chứng minh model tốt hơn.

`main.py` luôn ghi ra `output.mp4`, nên **đổi tên file sau lần chạy đầu**:

```powershell
# Model cũ
python main.py "video_moi.mp4"
Rename-Item output.mp4 output_model_cu.mp4

# Model mới (chỉ áp dụng trong cửa sổ terminal này)
$env:HELMET_WEIGHTS = "runs/detect/helmet_v2/weights/best.pt"
python main.py "video_moi.mp4"
Rename-Item output.mp4 output_model_moi.mp4
Remove-Item Env:HELMET_WEIGHTS
```

So sánh:
- **Số liệu:** dòng `Không đội mũ bảo hiểm: N` trong bảng tổng kết của hai lần chạy.
- **Video:** mở 2 file và xem các người **đội mũ đen** trước đây bị gắn "VI PHAM: KHONG MU" giờ còn bị gắn không.
- **Bỏ sót:** người **không đội mũ thật** vẫn phải bị bắt.

---

## 8. Đưa model mới vào web và quay lại model cũ

Chỉ làm khi **cả mục 6 và mục 7 đều ổn**:

1. Chép model vào thư mục `weights`:
   ```powershell
   Copy-Item runs\detect\helmet_v2\weights\best.pt weights\helmet_v2.pt
   ```
2. Thêm dòng sau vào file `.env` của backend:
   ```
   HELMET_WEIGHTS=weights/helmet_v2.pt
   ```
3. Khởi động lại backend (`Ctrl + C` rồi `uvicorn server:app --port 8000`).
4. Kiểm tra: phân tích một video. Trong phần thống kê mô hình của trang Phân tích video, dòng **Helmet Detector** phải hiện file `helmet_v2.pt`.

**Quay lại model cũ:** xoá hoặc comment dòng `HELMET_WEIGHTS` trong `.env`, rồi khởi động lại backend. File `weights/helmet_26s.pt` không bao giờ bị ghi đè.

Lưu ý: `.env` chỉ được nạp khi chạy **backend**. Khi chạy `main.py` hoặc các script trong `tools/`, đặt biến bằng `$env:HELMET_WEIGHTS = "..."` như ở mục 7, hoặc truyền `--weights`.

---

## 9. Các lần retrain sau

- **Cứ tiếp tục duyệt trên web:** dữ liệu trong `dataset/helmet_feedback/` tăng dần.
- **Mỗi lần retrain, train lại từ `weights/helmet_26s.pt`** với **toàn bộ** dữ liệu tích luỹ (mặc định của script). Đừng train chồng từ `helmet_v2.pt`, vì lỗi sẽ dồn qua các phiên bản.
- **Luôn so sánh với model đang chạy:**
  ```powershell
  python tools/helmet_eval.py --weights weights/helmet_v2.pt runs/detect/helmet_v3/weights/best.pt
  ```
- **Sao lưu `dataset/` định kỳ.** Thư mục này nằm trong `.gitignore`, nên không có trên Git.

---

## 10. Lỗi thường gặp

| Thông báo | Nguyên nhân / cách xử lý |
|---|---|
| `Chưa có ảnh nào đã gán nhãn…` | Chưa duyệt vi phạm mũ nào bằng nút Xác nhận / Có Đội Mũ, và `helmet_reviewed/` còn trống |
| Web không hiện dòng "dữ liệu huấn luyện" | Vi phạm phát hiện trước khi cập nhật code, hoặc không phải lỗi mũ. Hãy phân tích lại video. |
| `CUDA out of memory` | Thêm `--batch 4` (hoặc 2) |
| `UnicodeEncodeError: 'charmap'…` | Chạy `$env:PYTHONIOENCODING = "utf-8"` trước |
| `Lớp của bộ dữ liệu gốc … khác ['helmet', 'no_helmet']` | Bộ `--base-data` có lớp khác tên hoặc khác thứ tự |
| `⚠️ Bộ test còn rất ít ảnh` | Kết quả so sánh chưa đáng tin; gán nhãn thêm |
| `❌ CHƯA nên thay` | Đừng đổi `.env`. Kiểm tra lại nhãn (mục 3.2), thêm dữ liệu mũ đen, dùng `--base-data`. |
| `No such file … weights/helmet_26s.pt` | Đang chạy lệnh ngoài thư mục `traffic-ai-backend` |

---

## 11. Các file liên quan

| File | Vai trò |
|---|---|
| `utils/helmet_dataset.py` | Lưu mẫu thô; biến kết quả duyệt thành nhãn YOLO trong `dataset/helmet_feedback/` |
| `utils/pipeline.py` | Giữ khung hình sạch và khung đầu ở frame kết luận vi phạm mũ |
| `api/analyze.py` | `_save_violation`: lưu ảnh bằng chứng và mẫu thô |
| `api/violations.py` | `PATCH /api/v1/violations/{id}/status` (nhận `reason_code: "has_helmet"`), ghi nhãn khi duyệt |
| `api/cleanup.py` | Xoá vi phạm thì xoá cả mẫu thô (giữ `dataset/`) |
| `utils/helmet_violation.py` | Luật kết luận vi phạm (ngưỡng no_helmet 0.55, bỏ ca phân vân, cần ≥ 3 frame thấy rõ đầu ≥ 14px) |
| `models/helmet.py` | Nạp model theo biến `HELMET_WEIGHTS` (mặc định `weights/helmet_26s.pt`) |
| `tools/helmet_mine.py` | Khai thác khung hình có đầu đáng ngờ từ video |
| `tools/helmet_train.py` | Gộp dữ liệu, train tiếp, tự so sánh |
| `tools/helmet_eval.py` | So sánh các model trên cùng bộ test |
