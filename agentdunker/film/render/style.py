"""Shared look: bundled fonts, team colours and the broadcast-glass nameplate chip (REFERENCE_SPEC §2.2)."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = __import__("agentdunker.weights", fromlist=["x"]).fonts_dir()
CYAN = (90, 214, 255)
WHITE = (240, 246, 249)
NEUTRAL = (150, 165, 178)

# Bright accent (bar) per team; the badge uses a darkened version. Picked for contrast on dark glass.
TEAM_COLORS = {
    "ATL": (225, 68, 52), "BOS": (0, 160, 80), "BKN": (200, 200, 200), "CHA": (0, 150, 170), "CHI": (226, 30, 70),
    "CLE": (170, 30, 80), "DAL": (40, 110, 220), "DEN": (254, 197, 36), "DET": (210, 30, 60), "GSW": (255, 199, 44),
    "HOU": (226, 40, 70), "IND": (253, 187, 48), "LAC": (220, 30, 60), "LAL": (150, 90, 210), "MEM": (110, 140, 200),
    "MIA": (200, 20, 60), "MIL": (30, 150, 80), "MIN": (60, 140, 200), "NOP": (200, 170, 100), "NYK": (245, 132, 38),
    "OKC": (0, 125, 195), "ORL": (0, 125, 197), "PHI": (30, 100, 210), "PHX": (229, 95, 32), "POR": (224, 58, 62),
    "SAC": (120, 80, 200), "SAS": (196, 206, 211), "TOR": (206, 17, 65), "UTA": (60, 90, 200), "WAS": (227, 24, 55),
}
TEAM_ZH = {"ATL": "老鹰", "BOS": "凯尔特人", "BKN": "篮网", "CHA": "黄蜂", "CHI": "公牛", "CLE": "骑士", "DAL": "独行侠",
           "DEN": "掘金", "DET": "活塞", "GSW": "勇士", "HOU": "火箭", "IND": "步行者", "LAC": "快船", "LAL": "湖人",
           "MEM": "灰熊", "MIA": "热火", "MIL": "雄鹿", "MIN": "森林狼", "NOP": "鹈鹕", "NYK": "尼克斯", "OKC": "雷霆",
           "ORL": "魔术", "PHI": "76人", "PHX": "太阳", "POR": "开拓者", "SAC": "国王", "SAS": "马刺", "TOR": "猛龙",
           "UTA": "爵士", "WAS": "奇才"}


def team_color(abbr: str | None) -> tuple[int, int, int]:
    return TEAM_COLORS.get((abbr or "").upper(), NEUTRAL)


@lru_cache(maxsize=64)
def font(size: int, weight: int = 500, latin: bool = False) -> ImageFont.FreeTypeFont:
    """Noto Sans SC (variable) for Chinese, Barlow Condensed for numerals; system CJK font as fallback."""
    size = max(4, int(round(size)))
    if latin:
        p = FONTS / ("BarlowCondensed-BoldItalic.ttf" if weight >= 800 else "BarlowCondensed-Bold.ttf")
        if p.exists():
            return ImageFont.truetype(str(p), size)
    p = FONTS / "NotoSansSC-VF.ttf"
    if p.exists():
        f = ImageFont.truetype(str(p), size)
        try:
            f.set_variation_by_axes([weight])
        except (OSError, ValueError):
            pass
        return f
    for cand in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def name_chip(name: str, jersey: str, color: tuple[int, int, int], stat_caption: str = "", stat_value: str = "",
              scale: float = 1.0) -> Image.Image:
    """Glass plate: team bar | number badge | short name; optional second row (hero card stat)."""
    k = 2 * scale                                     # supersample ×2
    name_f, num_f = font(24 * k, 600), font(22 * k, 700, latin=True)
    cap_f, val_f = font(16 * k, 500), font(30 * k, 800, latin=True)
    m = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    pad, gap, row_h = round(10 * k), round(8 * k), round(40 * k)
    badge_w = max(round(30 * k), math.ceil(m.textlength(jersey, font=num_f)) + round(14 * k)) if jersey else 0
    name_w = math.ceil(m.textlength(name, font=name_f)) if name else 0
    width = pad + round(4 * k) + badge_w + (gap if jersey and name else 0) + name_w + pad
    height = row_h
    if stat_value:
        width = max(width, pad * 2 + math.ceil(m.textlength(stat_caption, font=cap_f)) + gap * 2
                    + math.ceil(m.textlength(stat_value, font=val_f)))
        height = row_h + round(36 * k)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, width - 1, height - 1), radius=round(7 * k), fill=(10, 20, 34, 214),
                        outline=(170, 206, 224, 70), width=max(1, round(k * .6)))
    d.line((pad, 1, width - pad, 1), fill=(200, 228, 240, 60), width=max(1, round(k * .6)))
    d.rounded_rectangle((round(3 * k), round(7 * k), round(7 * k), height - round(7 * k)), radius=round(2 * k),
                        fill=(*color, 245))
    x = pad + round(4 * k)
    if jersey:
        bh = round(26 * k)
        y0 = (row_h - bh) // 2
        dark = tuple(int(c * 0.55) for c in color)
        d.rounded_rectangle((x, y0, x + badge_w, y0 + bh), radius=round(5 * k), fill=(*dark, 235),
                            outline=(*color, 160), width=max(1, round(k * .6)))
        d.text((x + badge_w / 2, row_h / 2 + k), jersey, font=num_f, fill=(*WHITE, 255), anchor="mm")
        x += badge_w + gap
    if name:
        d.text((x, row_h / 2), name, font=name_f, fill=(*WHITE, 255), anchor="lm")
    if stat_value:
        d.line((pad, row_h, width - pad, row_h), fill=(190, 216, 230, 50), width=1)
        d.text((pad + round(4 * k), row_h + round(18 * k)), stat_caption, font=cap_f, fill=(170, 190, 205, 255), anchor="lm")
        d.text((width - pad, row_h + round(18 * k)), stat_value, font=val_f, fill=(*CYAN, 255), anchor="rm")
    return img.resize((math.ceil(width / 2), math.ceil(height / 2)), Image.Resampling.LANCZOS)


def with_alpha(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    return out
