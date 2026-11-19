# api/reports.py
# Xuất báo cáo vi phạm ra Excel (.xlsx) và PDF (A4 ngang, hỗ trợ tiếng Việt).
import datetime
import io
import os
from collections import Counter
from typing import cast
from xml.sax.saxutils import escape

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.styles.colors import Color
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.utils import get_column_letter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PDF_MAX_ROWS = 2000          # PDF quá dài vừa chậm vừa khó đọc -> giới hạn; Excel không giới hạn

COLUMNS = [
    ("STT", 6), ("Mã hồ sơ", 13), ("Biển số", 14), ("Loại vi phạm", 30), ("Loại xe", 11),
    ("Địa điểm", 28), ("Video", 24), ("Thời điểm", 18), ("Mốc video", 10), ("Tin cậy (%)", 11),
    ("Trạng thái", 13), ("Mô tả AI", 40),
]


def _row(i: int, v) -> list:
    task = v.task
    video = task.video if task else None
    return [
        i,
        v.violation_code,
        v.license_plate,
        v.violation_type.name if v.violation_type else "",
        v.vehicle_type or "",
        video.location.name if video and video.location else "",
        video.original_name if video else "",
        v.occurred_at.strftime("%d/%m/%Y %H:%M:%S") if v.occurred_at else "",
        f"{v.video_time_sec:.1f}s" if v.video_time_sec is not None else "",
        round(v.confidence_overall, 1) if v.confidence_overall is not None else "",
        v.status,
        v.ai_reason or "",
    ]


def summarize(violations) -> dict:
    by_type = Counter(v.violation_type.name if v.violation_type else "Khác" for v in violations)
    by_status = Counter(v.status for v in violations)
    by_location = Counter(
        (v.task.video.location.name if v.task and v.task.video and v.task.video.location else "Không rõ")
        for v in violations
    )
    return {"total": len(violations), "by_type": by_type, "by_status": by_status, "by_location": by_location}


def _filters_text(filters: dict) -> str:
    labels = {"search": "Từ khoá", "violation_type": "Loại vi phạm", "location": "Địa điểm", "status": "Trạng thái",
              "date_from": "Từ ngày", "date_to": "Đến ngày", "task_id": "Lần phân tích"}
    parts = [f"{labels[k]}: {v}" for k, v in filters.items() if v not in (None, "") and k in labels]
    return "; ".join(parts) if parts else "Tất cả hồ sơ"


# ─────────────────────────────── EXCEL ───────────────────────────────
def build_excel(violations, filters: dict, generated_by: str) -> bytes:
    wb = Workbook()
    ws = cast(Worksheet, wb.active)          # Workbook mới luôn có sẵn 1 sheet thường
    ws.title = "Vi phạm"
    head_fill = PatternFill("solid", fgColor=Color(rgb="FF1E3A8A"))
    head_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="CBD5E1")

    ws.append([h for h, _ in COLUMNS])
    for col, (_, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col)
        cell.fill, cell.font = head_fill, head_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col)].width = width
    for i, v in enumerate(violations, start=1):
        ws.append(_row(i, v))
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for c in row:
            c.border = Border(bottom=thin)
            c.alignment = Alignment(vertical="top", wrap_text=c.column == len(COLUMNS))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{max(1, ws.max_row)}"
    ws.row_dimensions[1].height = 30

    # Sheet tổng hợp
    s = summarize(violations)
    sm = cast(Worksheet, wb.create_sheet("Tổng hợp"))
    sm.column_dimensions["A"].width = 38
    sm.column_dimensions["B"].width = 14
    sm.column_dimensions["C"].width = 12
    title = sm.cell(row=1, column=1, value="BÁO CÁO VI PHẠM GIAO THÔNG — TRAFFIC AI")
    title.font = Font(bold=True, size=14, color="1E3A8A")
    sm.append(["Thời điểm xuất", datetime.datetime.now().strftime("%d/%m/%Y %H:%M")])
    sm.append(["Người xuất", generated_by])
    sm.append(["Bộ lọc", _filters_text(filters)])
    sm.append(["Tổng số hồ sơ", s["total"]])
    for label, counter in (("Theo loại vi phạm", s["by_type"]), ("Theo trạng thái", s["by_status"]), ("Theo địa điểm", s["by_location"])):
        sm.append([])
        r = sm.max_row + 1
        sm.append([label, "Số hồ sơ", "Tỷ lệ"])
        for row in sm.iter_rows(min_row=r, max_row=r):
            for c in row:
                c.font, c.fill = head_font, head_fill
        for name, n in counter.most_common():
            sm.append([name, n, f"{n / s['total'] * 100:.1f}%" if s["total"] else "0%"])
    for r in range(2, 6):
        sm.cell(row=r, column=1).font = Font(bold=True)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ─────────────────────────────── PDF ───────────────────────────────
_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
]
_font_names: tuple[str, str] | None = None


def _register_fonts() -> tuple[str, str]:
    """Font Unicode có dấu tiếng Việt (Helvetica mặc định của PDF không hiển thị được tiếng Việt)."""
    global _font_names
    if _font_names:
        return _font_names
    for regular, bold in _FONT_CANDIDATES:
        if os.path.isfile(regular):
            pdfmetrics.registerFont(TTFont("VN", regular))
            pdfmetrics.registerFont(TTFont("VN-Bold", bold if os.path.isfile(bold) else regular))
            _font_names = ("VN", "VN-Bold")
            return _font_names
    raise RuntimeError("Không tìm thấy font Unicode (Arial / Segoe UI / DejaVu) để xuất PDF tiếng Việt")


def build_pdf(violations, filters: dict, generated_by: str) -> bytes:
    font, bold = _register_fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=14 * mm, title="Báo cáo vi phạm giao thông — Traffic AI")
    st_title = ParagraphStyle("t", fontName=bold, fontSize=16, leading=20, textColor=colors.HexColor("#1E3A8A"))
    st_meta = ParagraphStyle("m", fontName=font, fontSize=9, leading=12, textColor=colors.HexColor("#334155"))
    st_h = ParagraphStyle("h", fontName=bold, fontSize=11, leading=14, spaceBefore=6, textColor=colors.HexColor("#0F172A"))
    st_cell = ParagraphStyle("c", fontName=font, fontSize=7.5, leading=9.5)

    s = summarize(violations)
    story = [
        Paragraph("BÁO CÁO VI PHẠM GIAO THÔNG", st_title),
        # Paragraph hiểu cú pháp XML (<b>, &...) -> dữ liệu người dùng nhập phải escape, không thì PDF lỗi
        Paragraph(f"Hệ thống Traffic AI · Xuất lúc {datetime.datetime.now():%d/%m/%Y %H:%M} · "
                  f"Người xuất: {escape(generated_by)}", st_meta),
        Paragraph(f"Bộ lọc: {escape(_filters_text(filters))} · Tổng số hồ sơ: <b>{s['total']}</b>", st_meta),
        Spacer(1, 6),
    ]

    # Bảng tổng hợp: 3 khối cạnh nhau
    def block(title, counter):
        rows = [[title, "SL"]] + [[Paragraph(escape(str(k)), st_cell), str(n)] for k, n in counter.most_common(8)]
        t = Table(rows, colWidths=[62 * mm, 14 * mm])
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), bold, 8), ("FONT", (0, 1), (-1, -1), font, 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t
    summary = Table([[block("Theo loại vi phạm", s["by_type"]), block("Theo trạng thái", s["by_status"]),
                      block("Theo địa điểm", s["by_location"])]], colWidths=[84 * mm] * 3)
    summary.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [summary, Spacer(1, 8)]

    # Bảng chi tiết (bỏ cột Video + Mô tả AI cho vừa khổ giấy)
    rows_src = violations[:PDF_MAX_ROWS]
    keep = [0, 1, 2, 3, 4, 5, 7, 9, 10]
    header = [COLUMNS[i][0] for i in keep]
    data = [header] + [[Paragraph(escape(str(row[k])), st_cell) for k in keep]
                     for row in (_row(i, v) for i, v in enumerate(rows_src, start=1))]
    widths = [10, 22, 24, 52, 18, 52, 30, 20, 22]
    detail = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), bold, 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F1F5F9")))
    detail.setStyle(TableStyle(style))
    story += [Paragraph("Danh sách hồ sơ vi phạm", st_h), detail]
    if len(violations) > PDF_MAX_ROWS:
        story.append(Paragraph(f"(PDF chỉ liệt kê {PDF_MAX_ROWS} hồ sơ đầu tiên trên tổng số {len(violations)}. "
                               "Dùng bản Excel để xem đầy đủ.)", st_meta))

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(12 * mm, 8 * mm, "Traffic AI — Báo cáo vi phạm giao thông")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 8 * mm, f"Trang {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
