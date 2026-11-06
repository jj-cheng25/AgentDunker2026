"""Chinese text helpers: number reading, CER, and fact guards (names / numbers)."""
from __future__ import annotations

import re

_DIG = "零一二三四五六七八九"
PUNCT = re.compile(r"[\s，。！？、；：,.!?;:“”\"'‘’（）()…—\-]+")
# Characters typical of transliterated foreign names. A run of >= 2 of them that is not
# an allowed name or a common word is treated as a possible invented name.
TRANSLIT = set("斯特克尔德姆森恩纳罗拉里卡布普夫兹维奥伊尼科莱塔沙希贾哈吉乔丹杜兰库詹弗格汤戴霍"
               "艾米诺佩利亚瓦蒂乌欧赫茨韦萨索基西谢巴马鲁阿洛莫利曼汉莱")
COMMON_TRANSLIT_WORDS = {"拉开", "拉起", "拉回", "卡位", "特别", "亚当", "马上", "伊始", "巴掌", "这里",
                         "那里", "哪里", "里面", "德比", "克制", "萨克", "罗列", "尼克", "利用", "顺利",
                         "犀利", "胜利", "锐利", "流利", "利落", "曼妙", "杜绝", "格外", "风格", "卡死"}


def int_zh(n: int) -> str:
    if n < 10:
        return _DIG[n]
    if n < 20:
        return "十" + (_DIG[n % 10] if n % 10 else "")
    if n < 100:
        return _DIG[n // 10] + "十" + (_DIG[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        tail = "" if rest == 0 else ("零" + _DIG[rest] if rest < 10 else ("一" + int_zh(rest) if rest < 20 else int_zh(rest)))
        return _DIG[n // 100] + "百" + tail
    return "".join(_DIG[int(c)] for c in str(n))


def num_zh(s: str) -> str:
    if "." in s:
        a, b = s.split(".", 1)
        return int_zh(int(a or 0)) + "点" + "".join(_DIG[int(c)] for c in b)
    return int_zh(int(s))


def reading(text: str) -> str:
    """Spoken form for TTS: 21.3% -> 百分之二十一点三, 7号 -> 七号."""
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: "百分之" + num_zh(m.group(1)), text)
    return re.sub(r"\d+(?:\.\d+)?", lambda m: num_zh(m.group(0)), text)


def norm(text: str) -> str:
    return PUNCT.sub("", reading(text)).replace("[pause", "")


def spoken_len(text: str) -> int:
    return len(re.sub(r"\[pause [\d.]+s\]", "", norm(text)))


def cer(ref: str, hyp: str) -> float:
    a, b = norm(ref), norm(hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def numbers_in(text: str) -> list[str]:
    """Arabic numbers plus Chinese numeric quantities that would state a fact (分/次/号/%/米/秒/英尺)."""
    found = re.findall(r"\d+(?:\.\d+)?", text)
    found += re.findall(r"百分之[零一二三四五六七八九十点]+", text)
    found += [m for m in re.findall(r"[两二三四五六七八九十百]+(?=分(?!之)|次|号|米|秒|英尺|成)", text)]
    return found


def bad_numbers(text: str, allowed: set[str]) -> list[str]:
    allowed = set(allowed) | {a.rstrip("%") for a in allowed}
    allowed_r = {reading(a) for a in allowed} | allowed
    return [n for n in numbers_in(text) if n not in allowed_r and reading(n) not in allowed_r]


def bad_names(text: str, allowed_names: set[str], roster: set[str]) -> list[str]:
    """Roster names that are not allowed, plus transliteration runs not covered by allowed names."""
    bad = [r for r in roster if r and r in text and not any(r in a or a in r for a in allowed_names if a)]
    masked = text
    for name in sorted(allowed_names, key=len, reverse=True):
        if name:
            masked = masked.replace(name, "　")
    for w in COMMON_TRANSLIT_WORDS:
        masked = masked.replace(w, "　")
    bad += re.findall("[" + "".join(TRANSLIT) + "]{2,}", masked)
    latin = re.search(r"[A-Za-z]{2,}", text)
    if latin:
        bad.append(latin.group(0))
    return list(dict.fromkeys(bad))


def _syllables(text: str) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:  # optional dependency; fall back to characters
        return list(norm(text))
    return lazy_pinyin(norm(text), style=Style.TONE3, neutral_tone_with_five=True)


def pinyin_cer(ref: str, hyp: str) -> float:
    """CER over toned syllables: exact homophones (强/墙, 协/斜) are not counted as errors."""
    a, b = _syllables(ref), _syllables(hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def edges_match(ref: str, hyp: str) -> bool:
    """First and last syllables of the read-back equal the script's (clipped onset / swallowed tail)."""
    a, b = _syllables(ref), _syllables(hyp)
    return bool(a) and bool(b) and a[0] == b[0] and a[-1] == b[-1]


def per_line_errors(lines: list[str], hyp: str) -> list[tuple[int, int]]:
    """Align one read-back of the whole track to the concatenated script.
    Returns (errors, length) per line, counted on the script side of the alignment."""
    refs = [norm(t) for t in lines]
    owner = [i for i, r in enumerate(refs) for _ in r]
    a, b = "".join(refs), norm(hyp)
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    errs = [0] * len(refs)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            errs[owner[i - 1]] += a[i - 1] != b[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            errs[owner[i - 1]] += 1
            i -= 1
        else:
            errs[owner[i - 1] if i > 0 else 0] += 1   # inserted characters count against the line they follow
            j -= 1
    return [(e, len(r)) for e, r in zip(errs, refs)]
