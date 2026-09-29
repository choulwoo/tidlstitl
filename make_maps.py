"""삼원가구 상차 경로 안내 지도 생성 스크립트.

원본 길찾기 캡처(images/route1.webp, images/route2.webp)에서
경유지 마커와 소요시간 말풍선을 지우고, 출발/도착을 크게 강조한
안내 지도(maps/경로1.png, maps/경로2.png)를 만든다.
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
ADDRESS = "경기도 포천시 송선로 150-34"
JIBUN = "(지번) 설운동 190-1"
SHOP_ADDRESS = "송선로 150-8"
SCALE = 2

GREEN = (22, 163, 74)
RED = (220, 38, 38)
NAVY = (15, 23, 42)
WHITE = (255, 255, 255)
ROUTE = (37, 99, 235)
ALLEY = (249, 115, 22)
YELLOW = (250, 204, 21)
DARK = (17, 24, 39)

ROUTES = [
    {
        "src": "images/route1.webp",
        "out": "maps/경로1.png",
        "title": "삼원가구 상차 경로 ①",
        "sub": "큰길 따라 남쪽 → ① 동교천 다리 건너기 직전 오른쪽 다리옆길 → 하천변 따라 서쪽 → ② 오른쪽 골목 · 약 2분",
        # 지울 영역: (x0, y0, x1, y1, 경로선 보존 여부)
        # 지울 영역 (x0, y0, x1, y1): 경유 마커, 소요시간 말풍선, 기존 핀
        "erase": [(600, 500, 636, 556), (402, 380, 466, 416),
                  (746, 486, 782, 552), (128, 134, 172, 180)],
        # 원본 경로선을 따라 찍은 좌표 (출발 → 도착)
        "path": [(757, 559), (723, 565), (698, 567), (673, 567), (648, 565), (623, 561),
                 (598, 551), (573, 533), (548, 510), (523, 491), (498, 476), (473, 464),
                 (440, 448), (415, 425), (390, 405), (365, 390), (340, 381), (300, 378),
                 (270, 381), (240, 378), (211, 377), (204, 365), (188, 340), (171, 315),
                 (158, 290), (154, 265), (162, 245), (170, 215), (172, 190), (166, 170)],
        "start_label_side": "left",
        "end_label_side": "right",
        # 골목으로 꺾는 지점 (path 인덱스)과 안내 박스 위치(확대 좌표 기준 좌상단)
        "turn": 20,
        "turn_box": (24, 800),
        "turn_title": "② 여기서 골목 진입!",
        "turn_text": "하천변 도로 끝에서 오른쪽 골목으로",
        # 큰길에서 동교천 다리 건너기 직전 오른쪽 다리옆길로 빠지는 지점 = 출발
        "start_lines": [("출발 · ① 다리 앞 우회전!", 30, GREEN),
                        ("큰길에서 동교천 다리 건너기 직전", 22, NAVY),
                        ("오른쪽 다리옆 하천변 길로 진입", 22, NAVY)],
        "tags": [((772, 572), "동교천 다리")],
        # 삼원가구 실제 위치(주소 지점)와 라벨 위치
        "shop": (117, 51),
        "shop_tag": (140, 22),
    },
    {
        "src": "images/route2.webp",
        "out": "maps/경로2.png",
        "title": "삼원가구 상차 경로 ②",
        "sub": "큰길 따라 북쪽 → ① 동교천 다리 건너 바로 오른쪽 다리옆길 → 하천변 따라 동쪽 → ② 왼쪽 골목 · 약 2분",
        "erase": [(376, 262, 412, 304), (322, 357, 390, 393),
                  (88, 504, 124, 556), (360, 106, 398, 152)],
        "path": [(106, 549), (201, 334), (230, 336), (260, 345), (290, 356), (320, 350),
                 (345, 342), (370, 340), (420, 344), (443, 349), (430, 330), (418, 312),
                 (405, 292), (392, 270), (385, 245), (390, 225), (400, 200), (403, 170),
                 (400, 150), (396, 142)],
        "start_label_side": "right",
        "end_label_side": "left",
        "turn": 9,
        "turn_box": (None, 750),
        "turn_title": "② 여기서 골목 진입!",
        "turn_text": "하천변 도로 끝에서 왼쪽 골목으로 크게 꺾기",
        # 큰길에서 동교천 다리를 건너자마자 오른쪽 다리옆길로 빠지는 지점
        "notes": [{
            "at": 1, "box": (16, 470),
            "lines": [("① 다리 건너자마자 우회전!", 32, DARK),
                      ("동교천 다리를 건너면 바로 오른쪽", 22, DARK),
                      ("다리옆 하천변 길로 진입", 22, DARK)],
        }],
        # 다리 위치 표시 (원본 좌표, 라벨 좌상단)
        "tags": [((110, 350), "동교천 다리")],
        "shop": (350, 25),
        "shop_tag": (214, 30),
    },
]


def route_pixels(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    blue = (b > 150) & (b - r > 120) & (g < 190)
    green = (g > 140) & (g - r > 90) & (g - b > 50)
    return blue | green


def font(size):
    return ImageFont.truetype(FONT, size)


def text_bold(draw, xy, text, size, fill, anchor="la", stroke=None):
    """wqy 폰트는 굵은체가 없어 같은 색 외곽선으로 굵게 표현한다."""
    sw = max(1, size // 22)
    if stroke:
        draw.text(xy, text, font=font(size), fill=stroke, anchor=anchor,
                  stroke_width=sw + 3, stroke_fill=stroke)
    draw.text(xy, text, font=font(size), fill=fill, anchor=anchor,
              stroke_width=sw, stroke_fill=fill)


def draw_pin(draw, tip, color, label):
    x, y = tip
    rad, h = 34, 78
    cy = y - h
    draw.ellipse((x - 10, y - 5, x + 10, y + 5), fill=(0, 0, 0))
    draw.polygon([(x - rad * 0.72, cy + rad * 0.7), (x + rad * 0.72, cy + rad * 0.7), (x, y)],
                 fill=color, outline=WHITE)
    draw.ellipse((x - rad - 4, cy - rad - 4, x + rad + 4, cy + rad + 4), fill=WHITE)
    draw.ellipse((x - rad, cy - rad, x + rad, cy + rad), fill=color)
    draw.polygon([(x - rad * 0.62, cy + rad * 0.78), (x + rad * 0.62, cy + rad * 0.78), (x, y - 4)],
                 fill=color)
    text_bold(draw, (x, cy), label, 24, WHITE, anchor="mm")
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=WHITE, outline=color, width=4)


def draw_callout(draw, tip, side, color, lines, img_w, img_h):
    """핀 옆에 설명 박스를 그린다. lines: [(text, size, color)]"""
    bw, bh = box_size(draw, lines)
    x, y = tip
    cy = y - 78
    if side == "right":
        x0 = x + 52
    else:
        x0 = x - 52 - bw
    y0 = cy - bh / 2
    x0 = min(max(x0, 12), img_w - bw - 12)
    y0 = min(max(y0, 12), img_h - bh - 12)
    draw_box(draw, (x0, y0), color, lines)


def box_size(draw, lines, pad=14):
    widths = [draw.textlength(t, font=font(s)) + s // 11 * 2 for t, s, _ in lines]
    heights = [s + 8 for _, s, _ in lines]
    return max(widths) + pad * 2, sum(heights) + pad * 2 - 8


def draw_box(draw, xy, color, lines, fill=WHITE, pad=14):
    x0, y0 = xy
    bw, bh = box_size(draw, lines, pad)
    draw.rounded_rectangle((x0 + 4, y0 + 5, x0 + bw + 4, y0 + bh + 5), 14, fill=(0, 0, 0, 110))
    draw.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 14, fill=fill, outline=color, width=5)
    ty = y0 + pad
    for t, s, c in lines:
        text_bold(draw, (x0 + pad, ty), t, s, c)
        ty += s + 8
    return bw, bh


def draw_route(draw, pts, color=ROUTE, width=16):
    """흰 테두리가 있는 경로선과 진행 방향 화살표를 그린다."""
    o = width + 10
    draw.line(pts, fill=WHITE, width=o, joint="curve")
    for p in (pts[0], pts[-1]):
        draw.ellipse((p[0] - o / 2, p[1] - o / 2, p[0] + o / 2, p[1] + o / 2), fill=WHITE)
    draw.line(pts, fill=color, width=width, joint="curve")
    for p in (pts[0], pts[-1]):
        draw.ellipse((p[0] - width / 2, p[1] - width / 2, p[0] + width / 2, p[1] + width / 2), fill=color)
    # 일정 간격마다 진행 방향 화살표
    gap, acc = 70, 35.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        t = gap - acc
        while t <= seg:
            cx, cy = x0 + ux * t, y0 + uy * t
            px, py = -uy, ux
            tip = (cx + ux * 6, cy + uy * 6)
            l = (cx - ux * 5 + px * 6, cy - uy * 5 + py * 6)
            r = (cx - ux * 5 - px * 6, cy - uy * 5 - py * 6)
            draw.line([l, tip, r], fill=WHITE, width=3, joint="curve")
            t += gap
        acc = (acc + seg) % gap


def draw_turn_note(draw, t, box, lines, img_w):
    bw, bh = box_size(draw, lines)
    bx, by = box
    if bx is None:
        bx = img_w - bw - 16
    near = (min(max(t[0], bx), bx + bw), min(max(t[1], by), by + bh))
    draw.line([t, near], fill=DARK, width=9)
    draw.line([t, near], fill=YELLOW, width=5)
    for r, c in ((40, DARK), (36, YELLOW), (26, DARK), (22, YELLOW)):
        draw.ellipse((t[0] - r, t[1] - r, t[0] + r, t[1] + r), outline=c, width=5)
    draw_box(draw, (bx, by), DARK, lines, fill=YELLOW)


def build(cfg):
    bgr = cv2.imread(cfg["src"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1) 원본 경로선, 경유 마커, 소요시간 말풍선, 기존 작은 핀 지우기
    route = route_pixels(rgb).astype(np.uint8) * 255
    route = cv2.morphologyEx(route, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.dilate(route, np.ones((7, 7), np.uint8))
    for x0, y0, x1, y1 in cfg["erase"]:
        mask[y0:y1, x0:x1] = 255
    bgr = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) 위성사진이 어두운 편이라 조금 밝게
    out = (rgb.astype(float) * 1.12 + 8).clip(0, 255).astype(np.uint8)

    # 3) 확대 후 경로/핀/설명 그리기
    img = Image.fromarray(out)
    w, h = img.size
    img = img.resize((w * SCALE, h * SCALE), Image.LANCZOS).convert("RGBA")
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    pts = [(x * SCALE, y * SCALE) for x, y in cfg["path"]]
    ti = cfg["turn"]
    draw_route(d, pts[:ti + 1])
    draw_route(d, pts[ti:], ALLEY, 20)   # 골목 구간은 주황색으로 굵게
    s, e = pts[0], pts[-1]

    # 꺾는 지점마다 노란 동그라미 + 안내 박스 + 연결선
    notes = cfg.get("notes", []) + [{
        "at": ti, "box": cfg["turn_box"],
        "lines": [(cfg.get("turn_title", "여기서 골목 진입!"), 32, DARK),
                  (cfg["turn_text"], 22, DARK),
                  ("주황색 길 따라 쭉 가면 도착", 22, (154, 52, 18))],
    }]
    for n in notes:
        draw_turn_note(d, pts[n["at"]], n["box"], n["lines"], W)
    for (x, y), label in cfg.get("tags", []):
        draw_box(d, (x * SCALE, y * SCALE), DARK, [(label, 20, DARK)], pad=8)
    draw_callout(d, s, cfg["start_label_side"], GREEN,
                 cfg.get("start_lines", [("출발", 30, GREEN), ("여기서 진입하세요", 22, NAVY)]), W, H)
    draw_callout(d, e, cfg["end_label_side"], RED,
                 [("도착 · 상차 장소", 30, RED), (ADDRESS, 22, NAVY), (JIBUN, 18, (71, 85, 105))], W, H)
    draw_pin(d, s, GREEN, "출발")
    draw_pin(d, e, RED, "도착")

    # 삼원가구 매장 위치 (images/samwon_location.png 기준으로 맞춘 좌표)
    shop = (cfg["shop"][0] * SCALE, cfg["shop"][1] * SCALE)
    tag = (cfg["shop_tag"][0] * SCALE, cfg["shop_tag"][1] * SCALE)
    lines = [("삼원가구", 22, RED), (SHOP_ADDRESS, 18, NAVY)]
    bw, bh = box_size(d, lines, pad=8)
    near = (min(max(shop[0], tag[0]), tag[0] + bw), min(max(shop[1], tag[1]), tag[1] + bh))
    d.line([shop, near], fill=WHITE, width=7)
    d.line([shop, near], fill=RED, width=3)
    draw_box(d, tag, RED, lines, pad=8)
    d.rectangle((shop[0] - 15, shop[1] - 15, shop[0] + 15, shop[1] + 15), fill=WHITE)
    d.rectangle((shop[0] - 11, shop[1] - 11, shop[0] + 11, shop[1] + 11), fill=RED)
    d.rectangle((shop[0] - 4, shop[1] - 4, shop[0] + 4, shop[1] + 4), fill=WHITE)
    img = Image.alpha_composite(img, layer)

    # 4) 상단 제목 띠
    head_h = 156
    canvas = Image.new("RGBA", (W, H + head_h), NAVY + (255,))
    canvas.paste(img, (0, head_h))
    d = ImageDraw.Draw(canvas)
    text_bold(d, (24, 20), cfg["title"], 38, WHITE)
    d.text((26, 74), cfg["sub"], font=font(21), fill=(203, 213, 225))
    lx, ly = 26, 122
    for color, label in ((ROUTE, "큰길"), (ALLEY, "골목 (여기로 들어가야 함)")):
        d.line([(lx, ly), (lx + 44, ly)], fill=WHITE, width=16)
        d.line([(lx + 2, ly), (lx + 42, ly)], fill=color, width=10)
        text_bold(d, (lx + 56, ly), label, 21, WHITE, anchor="lm")
        lx += 56 + d.textlength(label, font=font(21)) + 36
    canvas.convert("RGB").save(cfg["out"], optimize=True)
    print("saved", cfg["out"], canvas.size)


if __name__ == "__main__":
    for c in ROUTES:
        build(c)
