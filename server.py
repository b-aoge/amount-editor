# -*- coding: utf-8 -*-
"""
截图金额修改器 — V1
后端：
  - /api/analyze   OCR 检测全图文本，识别金额候选
  - /api/edit      支持两种定位：按目标原值自动定位 / 手动框选
                   -> 背景修复(inpaint) + 金额渲染(同字号/同色/同对齐) -> 返回图片

启动：uvicorn server:app --host 127.0.0.1 --port 8000
"""
import re
import os
import time
import uuid
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="截图金额修改器 V1")

# ---------------- OCR 引擎（懒加载） ----------------
_ocr_engine = None


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


# ---------------- 金额文本识别 ----------------
AMOUNT_RE = re.compile(
    r'^[¥￥$]\s*\d[\d,]*(\.\d+)?\s*$'   # 带货币符号
    r'|^\d[\d,]*(\.\d+)?\s*元\s*$'      # 带"元"
    r'|^\d{1,3}(,\d{3})+(\.\d+)?$'      # 千分位
    r'|^\d+(\.\d+)?$'                   # 纯数字
)


def is_amount(text):
    t = text.strip()
    return bool(AMOUNT_RE.match(t))


def normalize_amount(text):
    """归一化金额用于匹配：去货币符号/逗号/空格/元"""
    return re.sub(r'[¥￥$,\s元]', '', text.strip())


def ocr_boxes(img_bgr):
    """OCR 全图，返回 [{text, box(x,y,w,h), conf}]"""
    engine = get_ocr()
    result, _ = engine(img_bgr)
    items = []
    if not result:
        return items
    for line in result:
        pts, text, conf = line[0], line[1], line[2]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        items.append({
            "text": str(text),
            "box": [int(min(xs)), int(min(ys)),
                    int(max(xs) - min(xs)), int(max(ys) - min(ys))],
            "conf": float(conf),
        })
    return items

# ---------------- 指令解析 ----------------
INSTRUCTION_RE = re.compile(
    r'^(?:把|将|请把|请将|修改|更改)?\s*'
    r'(?:金额|价格|价钱|数字|¥|￥)?\s*'
    r'(?P<old>[\d,]+(?:\.\d+)?)\s*'
    r'(?:改成|改为|换成|变为|变成|改成是|改为是|变更为|调成|调整成|为|->|→|=>|⇒)\s*'
    r'[¥￥$]?\s*(?P<new>[\d,]+(?:\.\d+)?)\s*(?:元|块钱)?\s*$'
)


def parse_instruction(text):
    """解析「把 1,234.50 改成 5,678.90」类指令。返回 (old_normalized, new_value) 或 None。"""
    t = text.strip()
    m = INSTRUCTION_RE.match(t)
    if not m:
        return None
    old = normalize_amount(m.group("old"))
    new = normalize_amount(m.group("new"))
    if not old or not new:
        return None
    return old, new


def find_amount_box(items, target):
    """在 OCR 结果中按归一化原值查找金额框。返回 (box, text) 或 (None, None)。"""
    target = normalize_amount(target)
    for it in items:
        if is_amount(it["text"]) and normalize_amount(it["text"]) == target:
            return it["box"], it["text"]
    return None, None


def find_amount_boxes(items, target):
    """返回所有与 target 匹配的金额框列表 [(box, text), ...]"""
    target = normalize_amount(target)
    hits = []
    for it in items:
        if is_amount(it["text"]) and normalize_amount(it["text"]) == target:
            hits.append((it["box"], it["text"]))
    return hits


def join_ocr_texts(t1, t2):
    """拼接两个 OCR 片段，检测并去除重叠字符。"""
    a = t1.replace(" ", "")
    b = t2.replace(" ", "")
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, 0, -1):
        if a[-ov:] == b[:ov]:
            return a + b[ov:]
    return a + b


def merge_amount_items(items):
    """合并被 OCR 拆开的同行金额片段（如 '1234. ' + '50' -> '1234.50'）。"""
    merged = []
    used = set()
    for i, it in enumerate(items):
        if i in used:
            continue
        cur = dict(it)
        changed = True
        while changed:
            changed = False
            for j, it2 in enumerate(items):
                if j in used or j == i:
                    continue
                oy = (min(cur["box"][1] + cur["box"][3], it2["box"][1] + it2["box"][3])
                      - max(cur["box"][1], it2["box"][1]))
                if oy <= 0:
                    continue
                gap = it2["box"][0] - (cur["box"][0] + cur["box"][2])
                if it2["box"][0] >= cur["box"][0] and gap < 30:
                    new_text = join_ocr_texts(cur["text"], it2["text"])
                    if (is_amount(new_text)
                            or re.match(r"^[\d,]+\.?\d*$", new_text)):
                        nx0 = min(cur["box"][0], it2["box"][0])
                        nx1 = max(cur["box"][0] + cur["box"][2], it2["box"][0] + it2["box"][2])
                        ny0 = min(cur["box"][1], it2["box"][1])
                        ny1 = max(cur["box"][1] + cur["box"][3], it2["box"][1] + it2["box"][3])
                        cur = {
                            "text": new_text,
                            "box": [nx0, ny0, nx1 - nx0, ny1 - ny0],
                            "conf": min(cur["conf"], it2["conf"]),
                        }
                        used.add(j)
                        changed = True
        merged.append(cur)
    return merged


def decode_image(data):
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# ---------------- 图像处理 ----------------

# 字体（随仓库打包到 fonts/，Linux 部署环境无 Windows 字体）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS = {
    "Arial":          os.path.join(_BASE_DIR, "fonts", "arial.ttf"),
    "Arial Bold":     os.path.join(_BASE_DIR, "fonts", "arialbd.ttf"),
    "微软雅黑":         os.path.join(_BASE_DIR, "fonts", "msyh.ttc"),
    "微软雅黑 Bold":   os.path.join(_BASE_DIR, "fonts", "msyhbd.ttc"),
    "黑体 SimHei":     os.path.join(_BASE_DIR, "fonts", "simhei.ttf"),
    "宋体 SimSun":     os.path.join(_BASE_DIR, "fonts", "simsun.ttc"),
}
DEFAULT_FONT = "Arial"

# 响应头只能含 ASCII（HTTP header 为 latin-1），字体名的英文标识
FONT_ASCII = {
    "Arial": "Arial",
    "Arial Bold": "Arial_Bold",
    "微软雅黑": "Microsoft_YaHei",
    "微软雅黑 Bold": "Microsoft_YaHei_Bold",
    "黑体 SimHei": "SimHei",
    "宋体 SimSun": "SimSun",
}


def _render_mask(template, mask_bin):
    """模板与掩码的形态匹配度（Dice/F1）。"""
    tmpl = (template > 128).astype(np.uint8)
    m = (mask_bin > 128).astype(np.uint8)
    inter = int((tmpl & m).sum())
    total = int(tmpl.sum()) + int(m.sum())
    return (2.0 * inter / total) if total else 0.0


def match_font(mask_bin, text, text_height, align, x0, x1, y0, y1):
    """自动识别原数字字体：渲染 6 种候选字体模板，与主行掩码做像素级匹配。"""
    digits = normalize_amount(text)
    if not digits or text_height <= 2:
        return DEFAULT_FONT
    h, w = mask_bin.shape[:2]
    mask_w = x1 - x0 + 1
    mask_area = mask_bin[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    mask_density = int((mask_area > 128).sum()) / max(1, text_height * mask_w)
    k = np.ones((3, 3), np.uint8)
    mask_dil = cv2.dilate(mask_bin, k)
    scores = {}
    for name, path in FONTS.items():
        try:
            font_size = max(8, int(text_height * 1.2))
            font = None
            for _ in range(12):
                font = ImageFont.truetype(path, font_size)
                probe = Image.new("L", (w + 400, h + 400), 0)
                d = ImageDraw.Draw(probe)
                d.text((5, 5), digits, fill=255, font=font)
                arr = np.array(probe)
                rows = np.where(arr.any(axis=1))[0]
                if len(rows) == 0:
                    break
                cur_h = int(rows[-1] - rows[0] + 1)
                if abs(cur_h - text_height) <= 1:
                    break
                font_size = max(4, int(font_size * text_height / cur_h))
            if font is None:
                continue
            pad = max(w, h) + 120
            cw, ch = w + 2 * pad, h + 2 * pad
            canvas = Image.new("L", (cw, ch), 0)
            d = ImageDraw.Draw(canvas)
            d.text((pad, pad), digits, fill=255, font=font)
            arr = np.array(canvas)
            ys, xs = np.where(arr > 0)
            if len(xs) == 0:
                continue
            tw = int(xs.max() - xs.min() + 1)
            th = int(ys.max() - ys.min() + 1)
            cy = (y0 + y1) // 2
            exp_y0 = cy - th // 2
            if align == "right":
                exp_x0 = x1 - tw + 1
            elif align == "left":
                exp_x0 = x0
            else:
                exp_x0 = (x0 + x1) // 2 - tw // 2
            sx = int(exp_x0 - xs.min())
            sy = int(exp_y0 - ys.min())
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            shifted = cv2.warpAffine(arr, M, (w, h))
            shifted_dil = cv2.dilate(shifted, k)
            dice = _render_mask(shifted_dil, mask_dil)
            width_penalty = 1.0 - abs(tw - mask_w) / max(tw, mask_w)
            tmpl_density = int((shifted > 128).sum()) / max(1, th * tw)
            density_penalty = 1.0 - abs(tmpl_density - mask_density) / max(tmpl_density, mask_density)
            scores[name] = dice * width_penalty * (density_penalty ** 2)
        except Exception:
            continue
    if not scores:
        return DEFAULT_FONT
    normal_best = max(((n, s) for n, s in scores.items() if "Bold" not in n), key=lambda x: x[1])
    bold_best = max(((n, s) for n, s in scores.items() if "Bold" in n), key=lambda x: x[1])
    if mask_density < 0.42:
        return normal_best[0]
    if bold_best[1] > normal_best[1] * 1.35:
        return bold_best[0]
    return normal_best[0]


def analyze_region(region_bgr, strip_leading=False, strip_trailing=False):
    """分析框选区域：背景色、文字色、文字像素高度、对齐方式。返回 dict 或 None。"""
    h, w = region_bgr.shape[:2]
    if h < 3 or w < 3:
        return None
    edge = np.concatenate([
        region_bgr[0, :, :], region_bgr[-1, :, :],
        region_bgr[:, 0, :], region_bgr[:, -1, :],
    ], axis=0)
    bg = np.median(edge, axis=0)

    diff = np.abs(region_bgr.astype(int) - bg.astype(int)).max(axis=2)
    mask = (diff > 40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    row_counts = mask.sum(axis=1).astype(int)
    best_start, best_end, best_sum = 0, 0, -1
    start, cur = None, 0
    for i in range(len(row_counts)):
        if row_counts[i] > 0:
            if start is None:
                start = i
            cur += int(row_counts[i])
        else:
            if start is not None and cur > best_sum:
                best_sum, best_start, best_end = cur, start, i - 1
            start, cur = None, 0
    if start is not None and cur > best_sum:
        best_sum, best_start, best_end = cur, start, len(row_counts) - 1
    if best_sum <= 0:
        return None
    ys_full, xs = np.where(mask[best_start:best_end + 1] > 0)
    y0 = best_start + int(ys_full.min())
    y1 = best_start + int(ys_full.max())

    text_color = np.median(region_bgr[mask > 0], axis=0)
    text_height = int(y1 - y0 + 1)

    col = mask[y0:y1 + 1, :].sum(axis=0).astype(int)
    xl, xr = int(xs.min()), int(xs.max())
    sym_w = max(3, int(text_height * 1.05))
    stripped_left = stripped_right = False
    if xl < xr:
        seg_end = xl
        while seg_end < xr and col[seg_end] > 0:
            seg_end += 1
        seg_w = seg_end - xl
        gap_end = seg_end
        while gap_end <= xr and col[gap_end] == 0:
            gap_end += 1
        gap_w = gap_end - seg_end
        if (0 < seg_w < sym_w and gap_w >= max(4, int(text_height * 0.15))
                and gap_end <= xr + 1):
            nx = gap_end
            if nx <= xr:
                mask[y0:y1 + 1, :nx] = 0
                xl = nx
                stripped_left = True
    if strip_trailing and xr > xl:
        seg_start = xr
        while seg_start > xl and col[seg_start] > 0:
            seg_start -= 1
        seg_w = xr - seg_start
        if 0 < seg_w < sym_w and seg_start >= xl and col[seg_start] == 0:
            nx = seg_start
            while nx >= xl and col[nx] == 0:
                nx -= 1
            if nx >= xl:
                mask[y0:y1 + 1, nx + 1:] = 0
                xr = nx
                stripped_right = True

    left_gap = int(xl)
    right_gap = int(w - xr - 1)
    if stripped_left:
        align = "left"
    elif stripped_right:
        align = "right"
    elif left_gap > right_gap * 2.5:
        align = "right"
    elif right_gap > left_gap * 2.5:
        align = "left"
    else:
        align = "center"

    return {
        "mask": mask,
        "bg": bg,
        "text_color": text_color,
        "text_height": text_height,
        "align": align,
        "x0": int(xl), "x1": int(xr),
        "y0": int(y0), "y1": int(y1),
    }


def render_text(text, font_path, target_height, color_bgr, base_bgr, info, target_density=None):
    """在背景修复后的区域上渲染新值（stroke 矢量描边自动匹配粗细）。返回 (RGBA patch, off_x, off_y)。"""
    region_h, region_w = base_bgr.shape[:2]
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

    def render_once(stroke):
        font_size = max(8, int(target_height * 1.2))
        cur_h = 0
        for _ in range(12):
            font = ImageFont.truetype(font_path, font_size)
            probe = Image.new("L", (region_w + 600, region_h + 600), 0)
            ImageDraw.Draw(probe).text((5, 5), text, fill=255, font=font,
                                       stroke_width=stroke, stroke_fill=255)
            arr_p = np.array(probe)
            rows = np.where(arr_p.any(axis=1))[0]
            if len(rows) == 0:
                cur_h = 0
                break
            cur_h = rows[-1] - rows[0] + 1
            if abs(cur_h - target_height) <= 1:
                break
            font_size = max(4, int(font_size * target_height / cur_h))
        if cur_h <= 0:
            return None
        pad = max(region_w, region_h) + 200
        cw = region_w + 2 * pad
        ch = region_h + 2 * pad
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((pad, pad), text, font=font,
                                    fill=color_rgb + (255,),
                                    stroke_width=stroke, stroke_fill=color_rgb + (255,))
        arr = np.array(canvas)
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            return None
        dens = float((alpha / 255.0).sum()) / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        return arr, dens, pad

    pad = max(region_w, region_h) + 200
    cw = region_w + 2 * pad
    ch = region_h + 2 * pad
    if target_density:
        best = None
        for s in (0, 1, 2):
            r = render_once(s)
            if r is None:
                continue
            diff = abs(r[1] - target_density)
            if best is None or diff < best[0]:
                best = (diff, s, r[0])
        if best is None:
            return None, 0, 0
        _, best_s, arr = best
    else:
        r = render_once(0)
        if r is None:
            return None, 0, 0
        best_s, arr = 0, r[0]
    globals()["_last_w"] = best_s
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return None, 0, 0
    cur_x0, cur_y0, cur_x1, cur_y1 = xs.min(), ys.min(), xs.max(), ys.max()
    tw = int(cur_x1 - cur_x0 + 1)
    th = int(cur_y1 - cur_y0 + 1)

    cy = (info["y0"] + info["y1"]) // 2
    exp_y0 = cy - th // 2
    if info["align"] == "right":
        exp_x0 = info["x1"] - tw + 1
    elif info["align"] == "left":
        exp_x0 = info["x0"]
    else:
        cx = (info["x0"] + info["x1"]) // 2
        exp_x0 = cx - tw // 2

    shift_x = int(exp_x0 - (cur_x0 - pad))
    shift_y = int(exp_y0 - (cur_y0 - pad))

    margin = 2
    c0 = max(0, int(cur_x0) - margin)
    r0 = max(0, int(cur_y0) - margin)
    c1 = min(cw, int(cur_x1) + margin + 1)
    r1 = min(ch, int(cur_y1) + margin + 1)
    patch = arr[r0:r1, c0:c1]
    off_x = int(c0 - pad) + shift_x
    off_y = int(r0 - pad) + shift_y
    return patch, off_x, off_y


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    """OCR 检测全图文本，标记金额候选。返回 JSON。"""
    data = await image.read()
    img = decode_image(data)
    if img is None:
        return Response("无法解析图片", status_code=400)
    items = merge_amount_items(ocr_boxes(img))
    amounts = [it for it in items if is_amount(it["text"])]
    return {
        "texts": items,
        "amounts": amounts,
        "count": len(amounts),
    }


@app.post("/api/edit")
async def edit(
    image: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    w: int = Form(0),
    h: int = Form(0),
    new_value: str = Form(""),
    font: str = Form("auto"),
    old_value: str = Form(""),
):
    print("DEBUG_EDIT new_value=%r old_value=%r font=%r" % (new_value, old_value, font), flush=True)
    data = await image.read()
    img = decode_image(data)
    if img is None:
        return Response("无法解析图片", status_code=400)

    box = None
    ocr_calibrated = False
    matched_text = None
    if old_value.strip():
        items = merge_amount_items(ocr_boxes(img))
        hits = find_amount_boxes(items, old_value)
        print("DEBUG_EDIT old_match OCR=%s hits=%d" % (
            [(it["text"], it["box"]) for it in items], len(hits)), flush=True)
        if not hits:
            return Response(
                "未在图中找到目标金额「%s」，请检查原值或改用手动框选" % old_value,
                status_code=400,
            )
        if len(hits) > 1:
            return Response(
                "图中找到 %d 个「%s」金额，请在左侧金额列表中点选目标，或改用手动框选" % (len(hits), old_value),
                status_code=400,
            )
        box, matched_text = hits[0]
        box = [max(0, box[0] - 50), box[1], box[2] + 50, box[3]]
    elif w > 0 and h > 0:
        box = [x, y, w, h]
        ocr_calibrated = False
        try:
            items = merge_amount_items(ocr_boxes(img))
            bx0, by0, bw, bh = box
            best, best_ov = None, 0.0
            for it in items:
                if not is_amount(it["text"]):
                    continue
                ob = it["box"]
                ix0, iy0 = max(bx0, ob[0]), max(by0, ob[1])
                ix1, iy1 = min(bx0 + bw, ob[0] + ob[2]), min(by0 + bh, ob[1] + ob[3])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                ov = inter / max(1, bw * bh)
                if ov > best_ov:
                    best_ov, best = ov, it
            if best is not None and best_ov > 0.15:
                ob = best["box"]
                ny0 = min(by0, ob[1])
                ny1 = max(by0 + bh, ob[1] + ob[3])
                nx0 = max(bx0, ob[0])
                box = [nx0, ny0, bw - (nx0 - bx0), ny1 - ny0]
                ocr_calibrated = True
                matched_text = best["text"]
        except Exception:
            pass
    else:
        return Response("缺少定位信息：请框选数字或填写目标原值", status_code=400)

    if not new_value.strip():
        return Response("请填写新值", status_code=400)

    mtext = (matched_text or "").strip()
    strip_leading = bool(re.match(r"^[¥￥$]", mtext))
    strip_trailing = bool(re.search(r"元\s*$", mtext))

    H, W = img.shape[:2]
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(W, box[0] + box[2]), min(H, box[1] + box[3])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return Response("定位区域无效", status_code=400)

    region = img[y0:y1, x0:x1].copy()
    info = analyze_region(region, strip_leading, strip_trailing)
    if info is None:
        return Response("定位区域内未检测到文字，请改用手动框选紧贴数字", status_code=400)

    box_h = y1 - y0
    if ocr_calibrated:
        pad_top = max(8, int(box_h * 0.12))
        pad_bottom = max(8, int(box_h * 0.12))
    else:
        pad_top = max(16, int(box_h * 0.8))
        pad_bottom = max(10, int(box_h * 0.3))
    ex0 = max(0, y0 - pad_top)
    ex1 = min(H, y1 + pad_bottom)
    region = img[ex0:ex1, x0:x1].copy()
    info = analyze_region(region, strip_leading, strip_trailing)
    if info is None:
        return Response("定位区域内未检测到文字，请改用手动框选紧贴数字", status_code=400)
    y0, y1 = ex0, ex1

    mask_trim = info["mask"].copy()
    trim_pad = max(2, int(info["text_height"] * 0.15))
    y_lo = max(0, info["y0"] - trim_pad)
    y_hi = min(region.shape[0], info["y1"] + trim_pad + 1)
    mask_trim[:y_lo, :] = 0
    mask_trim[y_hi:, :] = 0
    dil = cv2.dilate(mask_trim, np.ones((5, 5), np.uint8))
    clean = cv2.inpaint(region, dil, 3, cv2.INPAINT_TELEA)

    font_name = (font or "auto").strip()
    if font_name == "auto":
        if matched_text:
            font_name = match_font(
                info["mask"], matched_text, info["text_height"],
                info["align"], info["x0"], info["x1"], info["y0"], info["y1"])
        else:
            font_name = DEFAULT_FONT
    font_path = FONTS.get(font_name, FONTS[DEFAULT_FONT])
    ma = info["mask"][max(0, info["y0"]):info["y1"] + 1, max(0, info["x0"]):info["x1"] + 1]
    mask_density = float((ma.astype(np.float32) / 255.0).sum()) / max(1, info["text_height"] * (info["x1"] - info["x0"] + 1))
    text_rgba, off_x, off_y = render_text(
        new_value, font_path, info["text_height"], info["text_color"], clean, info,
        target_density=mask_density)
    print("DEBUG_EDIT box=%s matched_text=%r strip=(%s,%s) region=%dx%d th=%d align=%s x0=%d x1=%d font=%s off=(%d,%d) md=%.4f w=%s" % (
        box, matched_text, strip_leading, strip_trailing,
        region.shape[1], region.shape[0], info["text_height"], info["align"],
        info["x0"], info["x1"], font_name, off_x, off_y, mask_density,
        globals().get("_last_w")), flush=True)

    out = img.copy()
    out[y0:y1, x0:x1] = clean
    ph, pw = text_rgba.shape[:2]
    dst_x0 = x0 + off_x
    dst_y0 = y0 + off_y
    src_x0 = max(0, -dst_x0)
    src_y0 = max(0, -dst_y0)
    dst_x0 = max(0, dst_x0)
    dst_y0 = max(0, dst_y0)
    w_eff = min(pw - src_x0, W - dst_x0)
    h_eff = min(ph - src_y0, H - dst_y0)
    if w_eff > 0 and h_eff > 0:
        patch = text_rgba[src_y0:src_y0 + h_eff, src_x0:src_x0 + w_eff]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        rgb = patch[:, :, :3].astype(np.float32)
        region_out = out[dst_y0:dst_y0 + h_eff, dst_x0:dst_x0 + w_eff].astype(np.float32)
        blended = rgb * alpha + region_out * (1.0 - alpha)
        out[dst_y0:dst_y0 + h_eff, dst_x0:dst_x0 + w_eff] = blended.astype(np.uint8)

    ok, buf = cv2.imencode(".png", out)
    png_bytes = buf.tobytes()

    out_dir = os.path.join("static", "output")
    os.makedirs(out_dir, exist_ok=True)
    fname = "result_%s_%s.png" % (time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6])
    with open(os.path.join(out_dir, fname), "wb") as f:
        f.write(png_bytes)
    headers = {
        "X-Result-Url": "/static/output/" + fname,
        "X-Used-Font": FONT_ASCII.get(font_name, font_name),
    }
    return Response(png_bytes, media_type="image/png", headers=headers)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
