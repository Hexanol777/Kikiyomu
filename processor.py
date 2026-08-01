import re
import hashlib
import io

KANJI_DUPLICATE_PATTERN = re.compile(r'([\u4e00-\u9fff])\1')

JP_RANGES = [
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FAF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3000, 0x303F),  # CJK Symbols & Punctuation
]


def is_valid_text(text, open_sign="「", close_sign="」"):
    if not text or not isinstance(text, str):
        return False

    if len(text.strip()) > 200:
        return False

    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]
    if any(ext in text.lower() for ext in image_exts):
        return False

    if text.startswith(open_sign) and text.endswith(close_sign):
        return False

    return any(
        any(start <= ord(c) <= end for start, end in JP_RANGES)
        for c in text[:20]
    )


def hash_image(image):
    """Generate an MD5 hash for an image to detect clipboard changes."""
    with io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        return hashlib.md5(buffer.getvalue()).hexdigest()


def remove_speaker_name(text, enabled):
    """Remove 【Speaker】 patterns from the start of a line (RPGMaker/WolfRPG)."""
    if not enabled:
        return text
    if text.startswith("【"):
        closing_index = text.find("】")
        if closing_index != -1 and closing_index != len(text) - 1:
            return text[closing_index + 1:].lstrip()
    return text


def remove_consecutive_kanji_duplicates(text):
    """Remove doubled kanji (e.g. 時時 → 時), caused by JS hooks in some RPGM games."""
    return KANJI_DUPLICATE_PATTERN.sub(r'\1', text)


def collapse_repetitions(text, enabled, min_len=1, max_len=30, threshold=2):
    """Remove substring-level and sentence-level repetitions from extracted text."""
    if not enabled:
        return text

    # Substring repetition removal
    for length in range(max_len, min_len - 1, -1):
        pattern = re.compile(rf'((.{{{length}}})\2{{{threshold - 1},}})')
        text = pattern.sub(r'\2', text)

    # Fast exact A+A duplication
    mid = len(text) // 2
    if text[:mid] == text[mid:]:
        text = text[:mid]

    # Sentence-level dedup
    sentences = re.split(r'(?<=[。！？\n])', text)
    seen = set()
    result = []
    for sentence in sentences:
        s = sentence.strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)

    return ''.join(result)


def word_filter(text, wordlist):
    """Remove each word in wordlist from text."""
    for word in wordlist:
        text = text.replace(word, "")
    return text