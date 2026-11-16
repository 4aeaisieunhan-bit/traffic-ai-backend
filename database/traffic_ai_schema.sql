-- =========================================================================================
-- Traffic AI - Script tạo Cơ sở dữ liệu cho Microsoft SQL Server (T-SQL)
-- Tương thích: SSMS 19.x & SQL Server Express 2022
--
-- CẤU TRÚC (9 bảng, ERD xem database/ERD.md):
--   users ─┬─< videos ──< analysis_tasks ─┬─< task_model_stats
--          │     ^                         └─< violations >── violation_types
--   locations ───┘                                 │
--          └───────────────< violation_reviews >───┘ (reviewer = users)
--
--   Chưa có dữ liệu camera -> nguồn dữ liệu là VIDEO upload. Khi có camera:
--   tạo bảng cameras(location_id) và thêm cột videos.camera_id (NULL được) - xem cuối file.
--
-- CÁCH CHẠY: mở file trong SSMS (Ctrl+O) -> F5.
--   * Script chỉ tạo cấu trúc + danh mục loại vi phạm (bảng locations để trống).
--   * Tài khoản admin (mật khẩu băm PBKDF2) và việc CHUYỂN DỮ LIỆU CŨ
--     được server.py tự làm khi khởi động (python server.py).
--   * Nếu DB còn bảng cấu trúc cũ, script đổi tên thành *_legacy (KHÔNG xóa dữ liệu).
-- =========================================================================================

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'traffic_ai_db')
    CREATE DATABASE [traffic_ai_db];
GO

USE [traffic_ai_db];
GO

-- -----------------------------------------------------------------------------------------
-- 0. GIỮ LẠI BẢNG CẤU TRÚC CŨ (đổi tên, không xóa) để server.py chuyển dữ liệu sang bảng mới
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.violations', N'U') IS NOT NULL AND COL_LENGTH(N'dbo.violations', N'task_id') IS NULL
   AND OBJECT_ID(N'dbo.violations_legacy', N'U') IS NULL
    EXEC sp_rename N'dbo.violations', N'violations_legacy';
GO

IF OBJECT_ID(N'dbo.analysis_tasks', N'U') IS NOT NULL AND COL_LENGTH(N'dbo.analysis_tasks', N'video_id') IS NULL
   AND OBJECT_ID(N'dbo.analysis_tasks_legacy', N'U') IS NULL
    EXEC sp_rename N'dbo.analysis_tasks', N'analysis_tasks_legacy';
GO

-- Bảng cameras của phiên bản rất cũ (không còn dùng)
IF OBJECT_ID(N'dbo.cameras', N'U') IS NOT NULL AND OBJECT_ID(N'dbo.cameras_legacy', N'U') IS NULL
    EXEC sp_rename N'dbo.cameras', N'cameras_legacy';
GO

-- -----------------------------------------------------------------------------------------
-- 1. USERS - tài khoản cán bộ
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.users', N'U') IS NULL
CREATE TABLE [dbo].[users] (
    [id]            INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_users] PRIMARY KEY,
    [email]         NVARCHAR(255) NOT NULL CONSTRAINT [UQ_users_email] UNIQUE,
    [password_hash] VARCHAR(255)  NOT NULL,                 -- pbkdf2_sha256$...
    [full_name]     NVARCHAR(100) NOT NULL,
    [role]          VARCHAR(20)   NOT NULL CONSTRAINT [DF_users_role] DEFAULT 'officer',
    [is_active]     BIT           NOT NULL CONSTRAINT [DF_users_is_active] DEFAULT 1,
    [created_at]    DATETIME2     NOT NULL CONSTRAINT [DF_users_created_at] DEFAULT SYSDATETIME(),
    CONSTRAINT [ck_users_role] CHECK ([role] IN ('admin', 'officer'))
);
GO

-- -----------------------------------------------------------------------------------------
-- 1b. USER_SESSIONS - phiên đăng nhập (chỉ lưu SHA-256 của token; đăng xuất = xóa dòng)
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.user_sessions', N'U') IS NULL
BEGIN
    CREATE TABLE [dbo].[user_sessions] (
        [id]         INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_user_sessions] PRIMARY KEY,
        [user_id]    INT NOT NULL CONSTRAINT [FK_user_sessions_users] FOREIGN KEY REFERENCES [dbo].[users]([id]),
        [token_hash] VARCHAR(64) NOT NULL CONSTRAINT [UQ_user_sessions_token_hash] UNIQUE,
        [created_at] DATETIME2 NOT NULL CONSTRAINT [DF_user_sessions_created_at] DEFAULT SYSDATETIME(),
        [expires_at] DATETIME2 NOT NULL
    );
    CREATE INDEX [ix_user_sessions_user_id] ON [dbo].[user_sessions]([user_id]);
    CREATE INDEX [ix_user_sessions_expires_at] ON [dbo].[user_sessions]([expires_at]);
END
GO

-- -----------------------------------------------------------------------------------------
-- 2. LOCATIONS - địa điểm giám sát
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.locations', N'U') IS NULL
CREATE TABLE [dbo].[locations] (
    [id]         INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_locations] PRIMARY KEY,
    [name]       NVARCHAR(255) NOT NULL CONSTRAINT [UQ_locations_name] UNIQUE,
    [address]    NVARCHAR(255) NULL,
    [latitude]   FLOAT NULL,
    [longitude]  FLOAT NULL,
    [created_at] DATETIME2 NOT NULL CONSTRAINT [DF_locations_created_at] DEFAULT SYSDATETIME()
);
GO

-- -----------------------------------------------------------------------------------------
-- 3. VIDEOS - file video đầu vào (nguồn dữ liệu thay cho camera)
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.videos', N'U') IS NULL
CREATE TABLE [dbo].[videos] (
    [id]            INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_videos] PRIMARY KEY,
    [original_name] NVARCHAR(255) NOT NULL,
    [stored_path]   NVARCHAR(500) NOT NULL,
    [file_size_mb]  FLOAT NULL,
    [width]         INT NULL,
    [height]        INT NULL,
    [fps]           FLOAT NULL,
    [total_frames]  INT NULL,
    [duration_sec]  FLOAT NULL,
    [location_id]   INT NULL CONSTRAINT [FK_videos_locations] FOREIGN KEY REFERENCES [dbo].[locations]([id]),
    [uploaded_by]   INT NULL CONSTRAINT [FK_videos_users] FOREIGN KEY REFERENCES [dbo].[users]([id]),
    [recorded_at]   DATETIME2 NULL,                         -- thời điểm quay (nếu biết)
    [uploaded_at]   DATETIME2 NOT NULL CONSTRAINT [DF_videos_uploaded_at] DEFAULT SYSDATETIME()
);
GO

-- -----------------------------------------------------------------------------------------
-- 4. ANALYSIS_TASKS - mỗi lần chạy pipeline AI trên 1 video
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.analysis_tasks', N'U') IS NULL
BEGIN
    CREATE TABLE [dbo].[analysis_tasks] (
        [id]                  VARCHAR(16) NOT NULL CONSTRAINT [PK_analysis_tasks] PRIMARY KEY,
        [video_id]            INT NOT NULL CONSTRAINT [FK_analysis_tasks_videos] FOREIGN KEY REFERENCES [dbo].[videos]([id]),
        [status]              VARCHAR(20) NOT NULL CONSTRAINT [DF_analysis_tasks_status] DEFAULT 'pending',
        [progress]            FLOAT NOT NULL CONSTRAINT [DF_analysis_tasks_progress] DEFAULT 0,
        [current_frame]       INT NOT NULL CONSTRAINT [DF_analysis_tasks_current_frame] DEFAULT 0,
        [total_frames]        INT NOT NULL CONSTRAINT [DF_analysis_tasks_total_frames] DEFAULT 0,
        [fps]                 FLOAT NULL,
        [violations_detected] INT NOT NULL CONSTRAINT [DF_analysis_tasks_violations] DEFAULT 0,
        [vehicles_scanned]    INT NOT NULL CONSTRAINT [DF_analysis_tasks_vehicles] DEFAULT 0,
        [legitimate_turns]    INT NOT NULL CONSTRAINT [DF_analysis_tasks_turns] DEFAULT 0,
        [output_video_path]   NVARCHAR(500) NULL,
        [processing_time_sec] FLOAT NULL,
        [error_message]       NVARCHAR(1000) NULL,
        [created_at]          DATETIME2 NOT NULL CONSTRAINT [DF_analysis_tasks_created_at] DEFAULT SYSDATETIME(),
        [started_at]          DATETIME2 NULL,
        [finished_at]         DATETIME2 NULL,
        CONSTRAINT [ck_analysis_tasks_status] CHECK ([status] IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
    );
    CREATE INDEX [ix_analysis_tasks_video_id] ON [dbo].[analysis_tasks]([video_id]);
    CREATE INDEX [ix_analysis_tasks_status] ON [dbo].[analysis_tasks]([status]);
END
GO

-- -----------------------------------------------------------------------------------------
-- 5. TASK_MODEL_STATS - thống kê thật của từng mô hình AI trong 1 lần phân tích
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.task_model_stats', N'U') IS NULL
CREATE TABLE [dbo].[task_model_stats] (
    [task_id]        VARCHAR(16)  NOT NULL CONSTRAINT [FK_task_model_stats_tasks] FOREIGN KEY REFERENCES [dbo].[analysis_tasks]([id]),
    [model_name]     NVARCHAR(50) NOT NULL,                 -- 'Vehicle', 'Traffic Light', ...
    [weight_file]    NVARCHAR(100) NOT NULL,
    [detections]     INT NOT NULL CONSTRAINT [DF_task_model_stats_detections] DEFAULT 0,
    [avg_confidence] FLOAT NULL,                            -- 0..1
    CONSTRAINT [PK_task_model_stats] PRIMARY KEY ([task_id], [model_name])
);
GO

-- -----------------------------------------------------------------------------------------
-- 6. VIOLATION_TYPES - danh mục loại vi phạm
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.violation_types', N'U') IS NULL
CREATE TABLE [dbo].[violation_types] (
    [id]          INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_violation_types] PRIMARY KEY,
    [code]        VARCHAR(30)   NOT NULL CONSTRAINT [UQ_violation_types_code] UNIQUE,
    [name]        NVARCHAR(100) NOT NULL CONSTRAINT [UQ_violation_types_name] UNIQUE,
    [description] NVARCHAR(255) NULL
);
GO

-- -----------------------------------------------------------------------------------------
-- 7. VIOLATIONS - hồ sơ vi phạm do AI phát hiện
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.violations', N'U') IS NULL
BEGIN
    CREATE TABLE [dbo].[violations] (
        [id]                   INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_violations] PRIMARY KEY,
        [violation_code]       NVARCHAR(50) NOT NULL CONSTRAINT [UQ_violations_code] UNIQUE,   -- 'VP-000001'
        [task_id]              VARCHAR(16) NOT NULL CONSTRAINT [FK_violations_tasks] FOREIGN KEY REFERENCES [dbo].[analysis_tasks]([id]),
        [violation_type_id]    INT NOT NULL CONSTRAINT [FK_violations_types] FOREIGN KEY REFERENCES [dbo].[violation_types]([id]),
        [track_id]             INT NULL,                    -- ID ByteTrack trong video
        [license_plate]        NVARCHAR(50) NOT NULL,
        [vehicle_type]         NVARCHAR(100) NULL,
        [vehicle_color]        NVARCHAR(50) NULL,
        [ai_reason]            NVARCHAR(255) NULL,          -- mô tả gốc từ thuật toán quỹ đạo
        [frame_index]          INT NULL,
        [video_time_sec]       FLOAT NULL,                  -- mốc trên video (giây)
        [occurred_at]          DATETIME2 NOT NULL CONSTRAINT [DF_violations_occurred_at] DEFAULT SYSDATETIME(),
        [confidence_overall]   FLOAT NULL,                  -- %
        [confidence_vehicle]   FLOAT NULL,
        [confidence_light]     FLOAT NULL,
        [confidence_plate]     FLOAT NULL,
        [confidence_violation] FLOAT NULL,
        [status]               NVARCHAR(50) NOT NULL CONSTRAINT [DF_violations_status] DEFAULT N'Chưa xử lý',
        [evidence_image]       NVARCHAR(255) NULL,
        [plate_crop_image]     NVARCHAR(255) NULL,
        [subject_crop_image]   NVARCHAR(255) NULL,          -- ảnh cận cảnh người vi phạm (lỗi mũ)
        [created_at]           DATETIME2 NOT NULL CONSTRAINT [DF_violations_created_at] DEFAULT SYSDATETIME(),
        CONSTRAINT [ck_violations_status] CHECK ([status] IN (N'Chưa xử lý', N'Đã xác nhận', N'Từ chối'))
    );
    CREATE INDEX [ix_violations_task_id] ON [dbo].[violations]([task_id]);
    CREATE INDEX [ix_violations_violation_type_id] ON [dbo].[violations]([violation_type_id]);
    CREATE INDEX [ix_violations_license_plate] ON [dbo].[violations]([license_plate]);
    CREATE INDEX [ix_violations_occurred_at] ON [dbo].[violations]([occurred_at]);
    CREATE INDEX [ix_violations_status] ON [dbo].[violations]([status]);
END
GO

-- -----------------------------------------------------------------------------------------
-- 8. VIOLATION_REVIEWS - lịch sử duyệt / từ chối hồ sơ
-- -----------------------------------------------------------------------------------------
IF OBJECT_ID(N'dbo.violation_reviews', N'U') IS NULL
BEGIN
    CREATE TABLE [dbo].[violation_reviews] (
        [id]           INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_violation_reviews] PRIMARY KEY,
        [violation_id] INT NOT NULL CONSTRAINT [FK_violation_reviews_violations] FOREIGN KEY REFERENCES [dbo].[violations]([id]),
        [reviewer_id]  INT NULL CONSTRAINT [FK_violation_reviews_users] FOREIGN KEY REFERENCES [dbo].[users]([id]),
        [old_status]   NVARCHAR(50) NOT NULL,
        [new_status]   NVARCHAR(50) NOT NULL,
        [note]         NVARCHAR(500) NULL,
        [reviewed_at]  DATETIME2 NOT NULL CONSTRAINT [DF_violation_reviews_reviewed_at] DEFAULT SYSDATETIME(),
        CONSTRAINT [ck_violation_reviews_old_status] CHECK ([old_status] IN (N'Chưa xử lý', N'Đã xác nhận', N'Từ chối')),
        CONSTRAINT [ck_violation_reviews_new_status] CHECK ([new_status] IN (N'Chưa xử lý', N'Đã xác nhận', N'Từ chối'))
    );
    CREATE INDEX [ix_violation_reviews_violation_id] ON [dbo].[violation_reviews]([violation_id]);
END
GO

-- -----------------------------------------------------------------------------------------
-- 9. DỮ LIỆU DANH MỤC
-- -----------------------------------------------------------------------------------------
MERGE [dbo].[violation_types] AS t
USING (VALUES
    ('RED_LIGHT', N'Vượt đèn đỏ',                     N'Đi thẳng / rẽ trái / rẽ phải khi đèn đỏ'),
    ('STOP_LINE', N'Không dừng đúng vạch khi đèn đỏ', N'Vượt qua vạch dừng và dừng lại trong giao lộ khi đèn đỏ'),
    ('NO_HELMET', N'Không đội mũ bảo hiểm',           N'Người điều khiển / ngồi sau xe máy không đội mũ bảo hiểm')
) AS s ([code], [name], [description])
ON t.[code] = s.[code]
WHEN NOT MATCHED THEN INSERT ([code], [name], [description]) VALUES (s.[code], s.[name], s.[description]);
GO

-- Bảng locations KHÔNG có dữ liệu cố định: địa điểm do người dùng quản lý qua API /api/v1/locations
-- (mọi lần phân tích video đều bắt buộc chọn một địa điểm).

PRINT N'✅ Đã tạo 9 bảng Traffic AI. Chạy "python server.py" để tạo tài khoản admin và chuyển dữ liệu cũ (nếu có). Không có dữ liệu demo.';
GO

-- -----------------------------------------------------------------------------------------
-- (TƯƠNG LAI) KHI CÓ DỮ LIỆU CAMERA - bỏ comment để thêm:
-- -----------------------------------------------------------------------------------------
-- CREATE TABLE [dbo].[cameras] (
--     [id]          INT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_cameras] PRIMARY KEY,
--     [location_id] INT NOT NULL CONSTRAINT [FK_cameras_locations] FOREIGN KEY REFERENCES [dbo].[locations]([id]),
--     [name]        NVARCHAR(100) NOT NULL,
--     [stream_url]  NVARCHAR(500) NULL,
--     [status]      VARCHAR(20) NOT NULL DEFAULT 'offline',
--     [created_at]  DATETIME2 NOT NULL DEFAULT SYSDATETIME()
-- );
-- ALTER TABLE [dbo].[videos] ADD [camera_id] INT NULL
--     CONSTRAINT [FK_videos_cameras] FOREIGN KEY REFERENCES [dbo].[cameras]([id]);
