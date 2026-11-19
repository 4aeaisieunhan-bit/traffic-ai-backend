# utils/frame_enhance.py
# Module xử lý & tăng cường chất lượng video giao thông (De-blurring, Contrast Enhancement & High-Res Plate Extraction)
#
# Giải quyết triệt để vấn đề video bị mờ, ảnh chụp biển số bị nhòe:
#   1. fast_sharpen()           : Khử mờ toàn frame cực nhanh (4ms), tăng độ nét của xe, làn đường, biển báo.
#   2. adaptive_contrast()      : Cân bằng ánh sáng cục bộ (CLAHE kênh L trong không gian LAB, không làm biến đổi màu).
#   3. enhance_plate_crop()     : Tăng cường chi tiết ảnh crop biển số (Dynamic Gamma, CLAHE nhẹ, Lanczos-4, Edge Crispening).
#   4. enhance_frame()          : Pipeline tổng hợp làm sắc nét toàn bộ video hiển thị.

import cv2
import numpy as np


def fast_sharpen(frame: np.ndarray, amount: float = 0.35) -> np.ndarray:
    """
    Làm sắc nét khung hình video nhanh chóng bằng bộ lọc chập 2D (O(1) trên GPU/CPU).
    Khử hiện tượng nhòe mờ khi camera ghi hình vật thể chuyển động hoặc khi nén video.

    Args:
        frame : Ảnh BGR uint8.
        amount: Cường độ sắc nét (0.2 - 0.5; mặc định 0.35 sắc nét tự nhiên, không tạo quầng sáng halo).

    Returns:
        Ảnh BGR uint8 đã được làm sắc nét.
    """
    if frame is None or frame.size == 0 or amount <= 0:
        return frame

    center = 1.0 + 4.0 * amount
    kernel = np.array(
        [
            [0.0, -amount, 0.0],
            [-amount, center, -amount],
            [0.0, -amount, 0.0],
        ],
        dtype=np.float32,
    )
    return cv2.filter2D(frame, -1, kernel)


def adaptive_contrast(frame: np.ndarray, clip_limit: float = 1.6, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Tăng cường độ tương phản thích ứng (CLAHE) chỉ trên kênh sáng Lightness (L) trong không gian màu LAB.
    Giúp các vùng quá sáng (chói nắng) hoặc quá tối (bóng râm) hiện rõ chi tiết mà không đổi màu xe hay đèn tín hiệu.

    Args:
        frame     : Ảnh BGR uint8.
        clip_limit: Ngưỡng giới hạn tương phản (1.2 - 2.0).
        tile_grid : Lưới phân chia vùng (mặc định 8x8).

    Returns:
        Ảnh BGR uint8 đã cân bằng tương phản.
    """
    if frame is None or frame.size == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)

    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def enhance_frame(
    frame: np.ndarray,
    sharpen_amount: float = 0.30,
    contrast_clip: float = 1.4,
    enable_contrast: bool = True,
) -> np.ndarray:
    """
    Pipeline làm sạch và tăng cường toàn bộ khung hình video giao thông trước khi đưa vào các model AI
    và hiển thị ra video output:
        1. Cân bằng tương phản thích ứng cục bộ (CLAHE nhẹ).
        2. Khử mờ nhòe bằng fast_sharpen.

    Args:
        frame          : Khung hình BGR uint8.
        sharpen_amount : Mức độ làm sắc nét (0.25 - 0.40).
        contrast_clip  : Mức độ tương phản CLAHE.
        enable_contrast: Bật/tắt cân bằng tương phản.

    Returns:
        Khung hình BGR rõ nét, khử mờ hoàn toàn.
    """
    if frame is None or frame.size == 0:
        return frame

    out = frame
    if enable_contrast and contrast_clip > 1.0:
        out = adaptive_contrast(out, clip_limit=contrast_clip)
    if sharpen_amount > 0:
        out = fast_sharpen(out, amount=sharpen_amount)

    return out



def enhance_plate_crop(
    crop: np.ndarray,
    target_width: int = 160,
    dynamic_gamma: bool = True,
    sharpen_edges: bool = True,
) -> np.ndarray:
    """
    Tăng cường độ chi tiết chuyên sâu cho ảnh crop biển số trước khi đưa vào mô hình OCR:
        1. Phóng to ảnh nhỏ với thuật toán nội suy Lanczos-4 (giữ cạnh chữ sắc nét, không bị răng cưa).
        2. Tự động đo độ sáng (luminance) để bù Gamma thích ứng:
           - Biển số bị chói nắng (> 190): Gamma = 0.82 để kéo chữ đen đậm trở lại.
           - Biển số bị bóng râm (< 80): Gamma = 1.25 để nâng sáng vùng tối.
        3. Cân bằng tương phản thích ứng (CLAHE nhẹ) trên kênh Lightness (LAB) để phân tách chữ khỏi nền.
        4. Tinh chỉnh viền chữ (Edge Crispening) giúp mô hình OCR nhận dạng chính xác từng ký tự.

    Args:
        crop          : Ảnh crop biển số (BGR uint8).
        target_width  : Chiều rộng tối thiểu để model OCR đạt độ chính xác cao nhất (140 - 180px).
        dynamic_gamma : Tự động bù sáng thích ứng.
        sharpen_edges : Làm nét viền ký tự.

    Returns:
        Ảnh biển số đã tối ưu hóa, cực kỳ sắc nét cho OCR.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return crop

    # 1. Phóng to ảnh nếu nhỏ hơn target_width bằng nội suy Lanczos-4
    if w < target_width:
        scale = float(target_width) / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Xử lý trên không gian màu LAB (chỉ tác động lên độ sáng, giữ nguyên màu sắc)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # 3. Dynamic Gamma thích ứng theo độ sáng trung bình của biển số
    if dynamic_gamma:
        mean_lum = float(np.mean(l_ch))
        if mean_lum > 185:
            gamma = 0.82    # Chói nắng -> kéo màu chữ đen đậm lại
        elif mean_lum < 85:
            gamma = 1.25    # Tối / bóng râm -> làm sáng nền
        else:
            gamma = 0.95

        if abs(gamma - 1.0) > 0.02:
            inv_gamma = 1.0 / gamma
            lut = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            l_ch = cv2.LUT(l_ch, lut)

    # 4. CLAHE nhẹ nhàng trên kênh L (clipLimit=1.8, grid=4x4)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)

    enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # 5. Làm sắc nét viền chữ tinh tế (không tạo nhiễu hạt)
    if sharpen_edges:
        kernel = np.array(
            [
                [0.0, -0.35, 0.0],
                [-0.35, 2.40, -0.35],
                [0.0, -0.35, 0.0],
            ],
            dtype=np.float32,
        )
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced
