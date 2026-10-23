"""
omr_core.py - OMR engine for the Voyage answer booklet.

The booklet ALWAYS has 60 questions (3 blocks x 20 rows x 5 options A-E).
The teacher decides how many of them are real questions (e.g. 40, 45, 50) - that
number and the marking scheme are passed in, so the sheet layout never changes.

Main entry points
    grade_script(image_bytes, key, n_questions)  -> ScriptResult
    parse_key_csv(bytes)                         -> {question: letter}
    parse_pasted_answers(text)                   -> [letters]
    build_scores_csv(rows) / build_overlay_zip(items) / save_outputs(...)
"""
import csv
import importlib.util
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------- settings
WARP_W, WARP_H = 1240, 1754            # A4 portrait, ~150 dpi
OPTIONS = "ABCDE"
TOTAL_QUESTIONS = 60
TOTAL_BUBBLES = TOTAL_QUESTIONS * 5
MIN_DROP = 25                          # a shaded bubble must be this much darker (0-255) than the row's blank ones
                                       # (real pencil marks measure 35+; shadows/smudges are usually under 10)
                                       # marks between MIN_DROP/2 and MIN_DROP are "faint": not counted, but flagged
MULTI_RATIO = 0.6                      # 2nd bubble this dark (vs the darkest) => double-shaded
BUBBLE_ROI = (0.08, 0.36, 0.60, 0.86)  # bubble area as page fractions (x0, y0, x1, y1)
NAME_BOX = (0.165, 0.24, 0.64, 0.298)  # name field as page fractions (x0, y0, x1, y1)
GOOD_MATCH = int(0.6 * TOTAL_BUBBLES)  # below this many matched bubbles the scan is flagged "low confidence"


class OMRError(Exception):
    """Raised when a sheet cannot be read; the message is safe to show to the teacher."""


@dataclass
class ScriptResult:
    answers: list          # 60 items: 'A'-'E', '' = blank, '*' = double-shaded
    score: int
    total: int
    answered: int          # questions (within the first `total`) that have a single clear mark
    blank: list            # question numbers left blank (within `total`)
    faint: list            # question numbers with only a faint mark - NOT counted, teacher should check
    multi: list            # question numbers with more than one shaded bubble
    confidence: float      # fraction of the 300 bubbles that were matched to the grid
    overlay: np.ndarray    # flattened page with the marking drawn on it (BGR)
    name_crop: np.ndarray  # cropped name field (BGR)


# ------------------------------------------------------ 1. page detection
def order_points(pts):
    pts = np.asarray(pts, dtype="float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")  # TL, TR, BR, BL


def find_page(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.3 * img.shape[0] * img.shape[1]:
        return None
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        return order_points(approx.reshape(4, 2))
    return order_points(cv2.boxPoints(cv2.minAreaRect(c)))


def flatten(img):
    corners = find_page(img)
    if corners is None:                                    # fall back to the whole image
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    dst = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (WARP_W, WARP_H))


# ------------------------------------------------------ 2. bubble grid
def find_bubbles(chan):
    blur = cv2.GaussianBlur(chan, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 8)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rx0, ry0, rx1, ry1 = BUBBLE_ROI
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        inside = rx0 * WARP_W <= x + w / 2 <= rx1 * WARP_W and ry0 * WARP_H <= y + h / 2 <= ry1 * WARP_H
        if inside and 12 <= w <= 34 and 10 <= h <= 30 and 0.9 <= w / h <= 1.9:
            boxes.append((x + w / 2, y + h / 2, w, h))
    if not boxes:
        return []
    mw, mh = np.median([b[2] for b in boxes]), np.median([b[3] for b in boxes])
    boxes = [b for b in boxes if abs(b[2] - mw) < 0.2 * mw and abs(b[3] - mh) < 0.2 * mh]
    out = []
    for b in sorted(boxes):
        if not any(abs(b[0] - o[0]) < 6 and abs(b[1] - o[1]) < 6 for o in out):
            out.append(b)
    return out


def fit_axis(vals, nominal, n, rel=0.15):
    """Find the evenly spaced positions v0 + k*pitch (k = 0..n-1) that best match the detections.
    Stray detections (ink, text, tick marks) are simply ignored. Returns (v0, pitch)."""
    vals = np.asarray(vals, float)
    best = (-1.0, 0.0, nominal)
    for pitch in np.linspace(nominal * (1 - rel), nominal * (1 + rel), 61):
        v0s = np.arange(vals.min() - pitch, vals.max() + 1, 1.0)[:, None]
        k = np.round((vals[None, :] - v0s) / pitch)
        err = np.abs(vals[None, :] - v0s - k * pitch)
        ok = (k >= 0) & (k < n) & (err < 0.2 * pitch)
        score = ok.sum(1) - 0.05 * (err * ok).sum(1) / pitch
        m = int(np.argmax(score))
        if score[m] > best[0]:
            best = (score[m], float(v0s[m, 0]), float(pitch))
    return best[1], best[2]


def split_blocks(boxes):
    """Split detections into the 3 question blocks using the biggest gaps between strong x-columns."""
    xs = np.array(sorted(b[0] for b in boxes))
    groups, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > 6:
            groups.append(cur)
            cur = []
        cur.append(x)
    groups.append(cur)
    centers = [float(np.mean(g)) for g in groups if len(g) >= 8]      # columns with real support
    if len(centers) < 3:
        raise OMRError("Could not locate the three question blocks. Is the whole sheet in the photo?")
    cuts = sorted(np.argsort(np.diff(centers))[-2:])
    edges = [-1e9] + [(centers[c] + centers[c + 1]) / 2 for c in cuts] + [1e9]
    return [[b for b in boxes if edges[k] <= b[0] < edges[k + 1]] for k in range(3)]


def fit_block(boxes, bw, per_block=20):
    """Fit a 20x5 grid to one block. Returns predicted centres (X, Y) and how many detections matched."""
    if len(boxes) < 20:
        raise OMRError("Too few bubbles found in one block. Try a sharper, better-lit photo.")
    xs = np.array([b[0] for b in boxes])
    ys = np.array([b[1] for b in boxes])
    x0, px = fit_axis(xs, 1.27 * bw, 5)
    y0, py = fit_axis(ys, 1.74 * bw, per_block)
    J, I = np.meshgrid(range(5), range(per_block))
    P = np.c_[np.ones(J.size), J.ravel(), I.ravel()]              # node k = row*5 + col
    cx, cy = np.array([x0, px, 0.0]), np.array([y0, 0.0, py])
    for _ in range(5):                                            # assign -> refit, so a small pitch error can't build up
        NX, NY = P @ cx, P @ cy
        d2 = (xs[:, None] - NX[None, :]) ** 2 + (ys[:, None] - NY[None, :]) ** 2
        node = d2.argmin(1)
        ok = d2[np.arange(len(xs)), node] < (0.3 * px) ** 2
        if ok.sum() < 30:
            raise OMRError("Could not line up the answer grid. Is this the right answer booklet?")
        A = np.c_[np.ones(ok.sum()), node[ok] % 5, node[ok] // 5]
        cx = np.linalg.lstsq(A, xs[ok], rcond=None)[0]            # x = a + b*col + c*row (allows slight rotation/shear)
        cy = np.linalg.lstsq(A, ys[ok], rcond=None)[0]
    # The paper can bend or curl, so let every row slide a little to sit on the bubbles actually found in it
    NX, NY = (P @ cx).reshape(per_block, 5), (P @ cy).reshape(per_block, 5)
    dx, dy = np.zeros(per_block), np.zeros(per_block)
    for _ in range(3):
        FX, FY = (NX + dx[:, None]).ravel(), (NY + dy[:, None]).ravel()
        d2 = (xs[:, None] - FX[None, :]) ** 2 + (ys[:, None] - FY[None, :]) ** 2
        node = d2.argmin(1)
        ok = d2[np.arange(len(xs)), node] < (0.3 * px) ** 2
        rx, ry = np.full(per_block, np.nan), np.full(per_block, np.nan)
        for r in range(per_block):
            sel = ok & (node // 5 == r)
            if sel.sum() >= 3:
                rx[r] = np.median(xs[sel] - NX.ravel()[node[sel]])
                ry[r] = np.median(ys[sel] - NY.ravel()[node[sel]])
        good = ~np.isnan(rx)
        if good.sum() >= 3:                                   # rows with too few detections borrow from their neighbours
            dx = np.interp(np.arange(per_block), np.flatnonzero(good), rx[good])
            dy = np.interp(np.arange(per_block), np.flatnonzero(good), ry[good])
    return NX + dx[:, None], NY + dy[:, None], int(ok.sum())


def build_grid(boxes, per_block=20):
    """Return (rows, matched). rows = 60 rows of 5 boxes in question order; matched = detections used."""
    if len(boxes) < 60:
        raise OMRError("Could not find the answer bubbles. Is the whole sheet in the photo?")
    bw = float(np.median([b[2] for b in boxes]))
    bh = float(np.median([b[3] for b in boxes]))
    rows, matched = [], 0
    for blk in split_blocks(boxes):
        X, Y, m = fit_block(blk, bw, per_block)
        matched += m
        for r in range(per_block):
            rows.append([(float(X[r, c]), float(Y[r, c]), bw, bh) for c in range(5)])
    return rows, matched


def locate_grid(img):
    page = flatten(img)
    # Red channel: pencil stays dark, but a teacher's red ink looks like blank paper
    chan = cv2.GaussianBlur(page[:, :, 2], (3, 3), 0)
    rows, matched = build_grid(find_bubbles(chan))
    return page, chan, rows, matched


def locate_with_rotation(img):
    """Try the photo as-is first; if the grid isn't found, try the other 3 orientations."""
    best, last_error = None, None
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        try:
            found = locate_grid(img if rot is None else cv2.rotate(img, rot))
        except OMRError as e:
            last_error = last_error or e
            continue
        if best is None or found[3] > best[3]:
            best = found
        if best[3] >= GOOD_MATCH:
            break
    if best is None or best[3] < TOTAL_BUBBLES * 0.3:
        raise last_error or OMRError("Could not find the answer grid in this image.")
    return best


# ------------------------------------------------------ 3. read the marks
def read_row(chan, row, min_drop=MIN_DROP):
    """Return (letter or None, status, marked_indices); status is OK, BLANK, MULTI or FAINT."""
    means = []
    for cx, cy, w, h in row:
        mx, my = int(w * 0.25), int(h * 0.25)             # inner area only, ignore the printed outline
        x0, y0 = int(cx - w / 2) + mx, int(cy - h / 2) + my
        x1, y1 = int(cx + w / 2) - mx, int(cy + h / 2) - my
        means.append(float(chan[y0:y1, x0:x1].mean()))
    drop = np.median(means) - np.array(means)              # compare against this row's blank bubbles
    top = drop.max()
    if top < min_drop / 2:
        return None, "BLANK", []
    if top < min_drop:                                     # something is there, but too light to trust
        return None, "FAINT", [int(drop.argmax())]
    marked = [i for i, d in enumerate(drop) if d >= max(min_drop, MULTI_RATIO * top)]
    if len(marked) > 1:
        return None, "MULTI", marked
    return OPTIONS[marked[0]], "OK", marked


# ------------------------------------------------------ 4. grade one script
def _draw_overlay(page, rows, reads, key, n_questions):
    out = page.copy()
    for q in range(1, n_questions + 1):
        ans, status, marked = reads[q - 1]
        correct = key.get(q)
        ok = ans is not None and ans == correct
        for i in marked:                                   # green = right, red = wrong, orange = double-shaded
            cx, cy, w, h = rows[q - 1][i]
            colour = ((0, 200, 0) if ok else (0, 140, 255) if status == "MULTI"      # green / orange
                      else (0, 215, 255) if status == "FAINT" else (0, 0, 255))      # yellow / red
            cv2.rectangle(out, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), colour, 2)
        if not ok and correct in OPTIONS:                  # blue ring = the right answer
            cx, cy, w, h = rows[q - 1][OPTIONS.index(correct)]
            cv2.circle(out, (int(cx), int(cy)), int(w * 0.8), (255, 0, 0), 2)
    return out


def grade_script(image_bytes, key, n_questions, min_drop=MIN_DROP):
    """Mark one photographed/scanned booklet.
    key: {question_number: 'A'..'E'} (needs entries for 1..n_questions)."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OMRError("This file could not be opened as an image.")
    page, chan, rows, matched = locate_with_rotation(img)

    reads = [read_row(chan, row, min_drop) for row in rows]
    answers = [r[0] or ("*" if r[1] == "MULTI" else "") for r in reads]
    scored = range(1, n_questions + 1)
    score = sum(1 for q in scored if reads[q - 1][0] is not None and reads[q - 1][0] == key.get(q))
    h, w = page.shape[:2]
    x0, y0, x1, y1 = NAME_BOX
    return ScriptResult(
        answers=answers,
        score=score,
        total=n_questions,
        answered=sum(1 for q in scored if reads[q - 1][1] == "OK"),
        blank=[q for q in scored if reads[q - 1][1] == "BLANK"],
        faint=[q for q in scored if reads[q - 1][1] == "FAINT"],
        multi=[q for q in scored if reads[q - 1][1] == "MULTI"],
        confidence=matched / TOTAL_BUBBLES,
        overlay=_draw_overlay(page, rows, reads, key, n_questions),
        name_crop=page[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].copy(),
    )


# ------------------------------------------------------ 5. marking scheme helpers
_Q_HEADERS = {"question", "q", "no", "no.", "number", "qn", "s/n", "sn", "question no", "question number", "q no"}
_A_HEADERS = {"answer", "ans", "key", "correct", "correct answer", "option", "answers"}


def _letter(cell):
    s = re.sub(r"[^A-Ea-e]", "", str(cell or ""))
    return s.upper() if len(s) == 1 else ""


def parse_key_csv(data):
    """Read a marking-scheme CSV. Accepts 'question,answer' with or without a header row, or a
    single column of answers in order. Returns {question_number: letter}."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("The file is empty.")
    first = [c.strip().lower() for c in rows[0]]
    has_header = not _letter(rows[0][-1]) or not first[-1] or len(first[-1]) > 1
    q_idx = a_idx = None
    if has_header:
        for i, name in enumerate(first):
            if name in _Q_HEADERS:
                q_idx = i
            if name in _A_HEADERS:
                a_idx = i
    ncols = max(len(r) for r in rows)
    if a_idx is None:
        a_idx = ncols - 1
    if q_idx is None and ncols > 1:
        q_idx = 0 if a_idx != 0 else None
    key = {}
    for n, r in enumerate(rows[1:] if has_header else rows, start=1):
        ans = _letter(r[a_idx]) if a_idx < len(r) else ""
        if not ans:
            raise ValueError(f"Row {n + (1 if has_header else 0)}: the answer must be a single letter A-E.")
        q = n
        if q_idx is not None and q_idx < len(r) and r[q_idx].strip().isdigit():
            q = int(r[q_idx].strip())
        key[q] = ans
    return key


def parse_pasted_answers(text):
    """'B C C A D' or 'BCCAD' or '1.B 2.C ...' -> ['B','C','C','A','D']"""
    text = re.sub(r"\d+", " ", text or "")
    return [c.upper() for c in re.findall(r"[A-Ea-e]", text)]


def missing_questions(key, n_questions):
    return [q for q in range(1, n_questions + 1) if q not in key]


def template_csv(n_questions):
    return "question,answer\n" + "".join(f"{q},\n" for q in range(1, n_questions + 1))


# ------------------------------------------------------ 6. name reading + outputs
def ocr_available():
    return all(importlib.util.find_spec(m) for m in ("transformers", "torch", "PIL"))


_ocr = {}


def ocr_name(crop_bgr):
    """Best-effort handwriting OCR (TrOCR). Needs: pip install transformers torch pillow. Always check the result."""
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if "model" not in _ocr:
        _ocr["proc"] = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _ocr["model"] = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    ids = _ocr["model"].generate(_ocr["proc"](images=img, return_tensors="pt").pixel_values)
    return _ocr["proc"].batch_decode(ids, skip_special_tokens=True)[0].strip()


def encode_jpg(img, quality=82):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def encode_png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def safe_filename(text, fallback="script"):
    s = re.sub(r"[^\w\- ]+", "", text or "", flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80] or fallback


def assign_filenames(pairs):
    """pairs: [(student_name, fallback)] -> unique file-name stems (John, John (2), ...)."""
    seen, out = {}, []
    for name, fallback in pairs:
        base = safe_filename(name, fallback)
        seen[base.lower()] = seen.get(base.lower(), 0) + 1
        out.append(base if seen[base.lower()] == 1 else f"{base} ({seen[base.lower()]})")
    return out


def build_scores_csv(rows):
    """rows: dicts with name, score, total, file. Returns UTF-8 (with BOM, so Excel opens it correctly)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Name", "Score", "Out of", "Percentage", "Script file"])
    for r in rows:
        pct = round(100 * r["score"] / r["total"], 1) if r["total"] else ""
        w.writerow([r["name"], r["score"], r["total"], pct, r["file"]])
    return buf.getvalue().encode("utf-8-sig")


def build_overlay_zip(items):
    """items: [(file_stem, jpg_bytes)] -> zip bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:       # JPEGs don't compress further
        for stem, jpg in items:
            z.writestr(f"{stem}.jpg", jpg)
    return buf.getvalue()


def save_outputs(folder, csv_bytes, overlay_items=None):
    """Write scores.csv (and, if given, overlays/<name>.jpg) into folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scores.csv").write_bytes(csv_bytes)
    if overlay_items is not None:
        od = folder / "overlays"
        od.mkdir(exist_ok=True)
        for old in od.glob("*.jpg"):                                # drop stale names from earlier saves
            old.unlink()
        for stem, jpg in overlay_items:
            (od / f"{stem}.jpg").write_bytes(jpg)
    return folder
