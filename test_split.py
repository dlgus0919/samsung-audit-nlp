import re
_KO_SUFFIXES = [
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "로",
    "이다", "이고", "이며", "하다", "하고", "하며", "한", "된", "되는",
    "있나요", "인가요", "인지", "나요", "인가", "했나요", "됐나요",
]
def _extract_keywords(query: str) -> list[str]:
    query = re.sub(r"[()\[\]~,]", " ", query)
    keywords = []
    for word in query.split():
        clean = word.strip("?!.,;:'\"")
        for suffix in sorted(_KO_SUFFIXES, key=len, reverse=True):
            if clean.endswith(suffix) and len(clean) - len(suffix) >= 2:
                clean = clean[:-len(suffix)]
                break
        if len(clean) >= 2:
            keywords.append(clean)
    return list(dict.fromkeys(keywords))

print(_extract_keywords("코로나19(COVID-19)로 인한 불확실성이 언급된 연도와 그 내용을 알려주세요."))
