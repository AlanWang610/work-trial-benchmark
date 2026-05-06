import tiktoken

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def get_encoding(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        _ENCODING_CACHE[model] = enc
    return _ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(get_encoding(model).encode(text))


def chunk(text: str, chunk_size_tokens: int, overlap: int = 0) -> list[str]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap < 0 or overlap >= chunk_size_tokens:
        raise ValueError("overlap must be in [0, chunk_size_tokens)")
    enc = get_encoding()
    tokens = enc.encode(text)
    if not tokens:
        return []
    step = chunk_size_tokens - overlap
    out: list[str] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size_tokens]
        out.append(enc.decode(window))
        i += step
    return out
