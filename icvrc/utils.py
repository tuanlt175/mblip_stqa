"""Các hàm utils."""
import re
import unicodedata as ud
import itertools

__specials__ = [r"==>", r"->", r"\.\.\.", r">>"]
__digit__ = r"\d+([\.,_]\d+)+"
__email__ = r"([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
__web__ = r"\w+://[^\s]+"
__word__ = r"\w+"
__non_word__ = r"[^\w\s]"
__abbreviations__ = [
    r"[A-ZĐ]+\.",
    r"[Tt]p\.",
    r"[Mm]r\.",
    r"[Mm]rs\.",
    r"[Mm]s\.",
    r"[Dd]r\.",
    r"[Tt]h[sS]\.",
]

__PATTERNS__ = (
    __abbreviations__
    + __specials__
    + [__web__, __email__, __digit__, __non_word__, __word__]
)
__REGEX_PATTERNS__ = "(" + "|".join(__PATTERNS__) + ")"


def sylabelize_word_special(text):
    """Hàm tách các dấu câu đặc biệt ra khỏi chữ cái.

    Hàm trả về list các từ và list dấu cách sau mỗi từ
    Ví dụ: "Lê Tuấn." -> ["Lê", "Tuấn", "."], [" ", "", ""]
    """
    text = ud.normalize("NFC", text)
    __tokens = [
        re.findall(__REGEX_PATTERNS__, word, re.UNICODE) for word in text.split()
    ]
    tokens = []
    spaces = []
    for words in __tokens:
        for word in words:
            word = split_alphanum(word[0]).split()
            tokens += word
            spaces += [" "] * (len(word) - 1) + [""]
        spaces[-1] = " "
    spaces[-1] = ""
    return tokens, spaces


def sylabelize(text):
    """Hàm tách các dấu câu đặc biệt ra khỏi chữ cái.

    Ví dụ: "Tuấn." -> "Tuấn ."
    """
    text = ud.normalize("NFC", text)
    tokens = re.findall(__REGEX_PATTERNS__, text, re.UNICODE)

    return " ".join([split_alphanum(token[0]) for token in tokens])


def normalize(text):
    """Tiền xử lý dữ liệu."""
    text = ud.normalize("NFC", text)
    text = " ".join(text.split())
    text = text.replace("–", "-")
    text = "".join([char for char in text if ord(char) < 8000])
    return text


def normalize_unicode(text):
    """Chuẩn hóa unicode cho dữ liệu text."""
    return ud.normalize("NFC", text)


RE_AL_NUM = re.compile(r"([^\d]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([^\d]+)", flags=re.UNICODE)
ACCENTS = r"[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]"

ALL_CHAR_ACCENTS = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"


def remove_special_char(text):
    """Xóa các ký tự đặc biệt trong câu, thay thể bằng khoảng trắng."""
    chars = [
        char if char.isascii() or char in ALL_CHAR_ACCENTS else " " for char in text
    ]
    text = "".join(chars)
    text = " ".join(text.split())
    text = " ".join(text.replace("( )", " ").replace("[ ]", " ").split())
    return text


def split_alphanum(s):
    """Nếu từ s chứa dấu câu, tách riêng biệt số và chữ cái ra, nếu không thì để nguyên.

    Ví dụ:
        - "GPRS5", "a12", "3g", "lethanhtuan12345" -> giữ nguyên chuỗi
        - "Tuấn18" -> "Tuấn 18", "Hà_Nội29" -> "Hà_Nội 29"
    Args:
        s (str): Từ (word/token)

    Returns:
        str: một chuỗi mới tương ứng với từ đầu vào, nhưng được tách riêng biệt số và chữ cái
    """
    if re.search(ACCENTS, s):
        s = normalize_unicode(s)
        s = RE_AL_NUM.sub(r"\1 \2", s)
        return RE_NUM_AL.sub(r"\1 \2", s)
    else:
        return s


def remove_accents(s):
    """Xóa dấu của một câu.

    Ví dụ: Xóa dấu -> Xoa dau
    """
    s = re.sub("Đ", "D", s)
    s = re.sub("đ", "d", s)
    s = ud.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    return s


def fix_sticky_words(text):
    """Sửa lỗi cuối của câu này bị dính với từ đầu của câu kia."""

    def _check_sticky_words(word):
        if remove_accents(word) == word:
            return False
        if word.islower() or word.istitle() or word.isupper():
            return False
        return True

    def _refine_word(word):
        if not _check_sticky_words(word):
            return word
        else:
            for i, char in enumerate(word):
                if char.islower():
                    continue
                else:
                    return word[:i] + " . " + word[i:]
        return word

    text = " ".join([_refine_word(word) for word in text.split()])
    return text


def contain_alnum(word):
    """Xác định xem một từ có chứa chữ hoặc số không."""
    for char in word:
        if char.isalnum():
            return True
    return False


def strip_punctuations(text):
    """Xóa các dấu câu ở hai đầu câu."""
    text = re.sub(r"[ \.,!\?]*(?=$)", "", text)
    text = re.sub(r"(?<=^)[ \.,!\?]*", "", text)
    if len(text) < 2:
        return text
    for x, y in ["()", "[]", "{}", "\"\"", "''", "``"]:
        if text[0] == x and text[-1] == y:
            text = text[1:-1].strip()
    return text


def preprocess_text(text):
    """Tiền xử lý dữ liệu text từ tin nhắn người dùng."""
    # chuẩn hóa unicode
    text = ud.normalize("NFC", text)
    # Bỏ đi mấy icon hay kí tự đặc biệt
    text = "".join([char if ord(char) < 8000 else " " for char in text])
    # Phân tách dấu câu ra khỏi từ
    text = sylabelize(text)
    # lỗi cuối của câu này bị dính với từ đầu của câu kia
    text = fix_sticky_words(text)
    # lower case
    text = text.lower()
    # xóa các dấu câu ở hai đầu câu
    text = strip_punctuations(text)
    text = " ".join(text.split())
    return text


def flatten_list(a_list):
    """Làm phẳng một danh sách hai chiều không đồng nhất.

    ví dụ:
    [[1,2,3,4], [1,3], [5], [6,7,8]] --> [1, 2, 3, 4, 1, 3, 5, 6, 7, 8]
    """
    return list(itertools.chain.from_iterable(a_list))


def trim_question_mark(question: str) -> str:
    question = question.rstrip(" ?").rstrip().rstrip("?")
    return question


def sentence_tokenize(context):
    """Tách câu."""
    def refine_sentences(sentences):
        if len(sentences) > 1:
            for i, s in enumerate(sentences[:-1]):
                if re.fullmatch(r"[0-9a-zA-Z]{1,4}\.", s):
                    sentences[i + 1] = s + " " + sentences[i + 1]
                    sentences[i] = ""

        sentences = [s for s in sentences if s != ""]

        if len(sentences) > 1:
            for i in range(1, len(sentences)):
                if len(sentences[i]) < 5:
                    sentences[i - 1] = sentences[i - 1] + " " + sentences[i]
                    sentences[i] = ""
        return [s for s in sentences if s != ""]

    vi_alpha = f"[A-Z{ALL_CHAR_ACCENTS.upper()}]"
    vi_alpha_lower = vi_alpha.lower()
    split_regex = f'(?<![A-Z][a-z]\.)(?<!{vi_alpha}\.)'
    split_regex += r'(?<=\.)\s'
    split_regex += f'(?![0-9])(?!{vi_alpha_lower})'
    split_regex += r'(?![\(])(?![\{])(?![\[])'
    contexts = [re.split(split_regex, s) for s in context.splitlines()]
    contexts = [refine_sentences(sentences) for sentences in contexts]
    contexts = flatten_list(contexts)
    return contexts
