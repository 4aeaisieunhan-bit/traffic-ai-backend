# Traffic AI – Thiết kế Cơ sở dữ liệu

Chưa có dữ liệu camera, nên **nguồn dữ liệu là video được upload** (`videos`). Bảng `cameras` được để sẵn chỗ cho sau này (nét đứt trong sơ đồ): khi có camera chỉ cần tạo bảng `cameras` và thêm cột `videos.camera_id` (cho phép NULL), các bảng khác giữ nguyên.

## Sơ đồ ERD

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : "đăng nhập"
    USERS ||--o{ VIDEOS : "upload"
    USERS |o--o{ VIOLATION_REVIEWS : "duyệt"
    LOCATIONS |o--o{ VIDEOS : "quay tại"
    LOCATIONS ||--o{ CAMERAS : "lắp đặt (tương lai)"
    CAMERAS |o..o{ VIDEOS : "ghi hình (tương lai)"
    VIDEOS ||--o{ ANALYSIS_TASKS : "được phân tích"
    ANALYSIS_TASKS ||--o{ TASK_MODEL_STATS : "thống kê mô hình"
    ANALYSIS_TASKS ||--o{ VIOLATIONS : "phát hiện"
    VIOLATION_TYPES ||--o{ VIOLATIONS : "phân loại"
    VIOLATIONS ||--o{ VIOLATION_REVIEWS : "lịch sử duyệt"

    USERS {
        int id PK
        nvarchar email UK
        varchar password_hash
        nvarchar full_name
        varchar role "admin | officer"
        bit is_active
        datetime2 created_at
    }
    USER_SESSIONS {
        int id PK
        int user_id FK
        varchar token_hash UK "SHA-256 của token"
        datetime2 created_at
        datetime2 expires_at
    }
    LOCATIONS {
        int id PK
        nvarchar name UK
        nvarchar address
        float latitude
        float longitude
    }
    CAMERAS {
        int id PK
        int location_id FK
        nvarchar name
        nvarchar stream_url
        varchar status
    }
    VIDEOS {
        int id PK
        nvarchar original_name
        nvarchar stored_path
        float file_size_mb
        int width
        int height
        float fps
        int total_frames
        float duration_sec
        int location_id FK "NULL được"
        int uploaded_by FK "NULL được"
        datetime2 recorded_at "NULL được"
        datetime2 uploaded_at
    }
    ANALYSIS_TASKS {
        varchar id PK "task_id 8 ký tự"
        int video_id FK
        varchar status "pending | processing | completed | failed"
        float progress
        int current_frame
        int total_frames
        float fps
        int violations_detected
        int vehicles_scanned
        int legitimate_turns
        nvarchar output_video_path
        float processing_time_sec
        nvarchar error_message
        datetime2 created_at
        datetime2 started_at
        datetime2 finished_at
    }
    TASK_MODEL_STATS {
        varchar task_id PK,FK
        nvarchar model_name PK
        nvarchar weight_file
        int detections
        float avg_confidence "0..1"
    }
    VIOLATION_TYPES {
        int id PK
        varchar code UK "RED_LIGHT | STOP_LINE | NO_HELMET"
        nvarchar name UK
        nvarchar description
    }
    VIOLATIONS {
        int id PK
        nvarchar violation_code UK "VP-000001"
        varchar task_id FK
        int violation_type_id FK
        int track_id
        nvarchar license_plate
        nvarchar vehicle_type
        nvarchar vehicle_color
        nvarchar ai_reason
        int frame_index
        float video_time_sec
        datetime2 occurred_at
        float confidence_overall
        float confidence_vehicle
        float confidence_light
        float confidence_plate
        float confidence_violation
        nvarchar status "Chưa xử lý | Đã xác nhận | Từ chối"
        nvarchar evidence_image
        nvarchar plate_crop_image
        nvarchar subject_crop_image
        datetime2 created_at
    }
    VIOLATION_REVIEWS {
        int id PK
        int violation_id FK
        int reviewer_id FK "NULL được"
        nvarchar old_status
        nvarchar new_status
        nvarchar note
        datetime2 reviewed_at
    }
```

## Giải thích quan hệ

| Quan hệ | Kiểu | Ý nghĩa |
|---|---|---|
| `users` – `user_sessions` | 1 – n | Một tài khoản đăng nhập trên nhiều thiết bị; đăng xuất thì xóa phiên |
| `users` – `videos` | 1 – n | Một cán bộ upload nhiều video |
| `locations` – `videos` | 1 – n | Nhiều video quay tại cùng một địa điểm |
| `videos` – `analysis_tasks` | 1 – n | Một video có thể phân tích nhiều lần (ví dụ khi đổi tham số hoặc mô hình) |
| `analysis_tasks` – `task_model_stats` | 1 – n | Mỗi lần phân tích lưu số liệu của 6 mô hình AI |
| `analysis_tasks` – `violations` | 1 – n | Mỗi lần phân tích phát hiện nhiều vi phạm |
| `violation_types` – `violations` | 1 – n | Danh mục chuẩn hóa loại vi phạm |
| `violations` – `violation_reviews` | 1 – n | Mỗi lần đổi trạng thái hồ sơ được ghi lại |
| `users` – `violation_reviews` | 1 – n | Cán bộ nào đã duyệt |

## Quy tắc thiết kế

- **Không lưu dữ liệu tính được:** giờ, ngày và mốc video hiển thị (`time_str`, `date_str`, `video_time`) được tính từ `occurred_at` và `video_time_sec` khi trả API.
- **Thời điểm vi phạm:** `occurred_at = videos.recorded_at + video_time_sec`. Nếu không biết thời điểm quay thì dùng `uploaded_at`.
- **Độ tin cậy NULL được:** cột để NULL khi mô hình không cung cấp điểm, thay vì điền số giả.
- **Toàn vẹn dữ liệu:** `status` và `role` có ràng buộc `CHECK`, các khóa ngoại đều được khai báo.
- **Không mất dữ liệu cũ:** bảng cấu trúc cũ được đổi tên thành `*_legacy`. `server.py` chuyển dữ liệu sang bảng mới khi khởi động.

## Vẽ ERD

- **VS Code:** cài extension *Markdown Preview Mermaid Support* rồi mở file này ở chế độ Preview.
- **SSMS:** chọn `traffic_ai_db` → *Database Diagrams* → *New Database Diagram* → chọn 9 bảng. SSMS tự vẽ quan hệ từ các khóa ngoại.
- **Online:** dán khối mermaid ở trên vào https://mermaid.live để xuất PNG/SVG.
