"""Carga del documento fuente y segmentación en chunks.

Estrategia elegida: chunking por párrafos con tamaño objetivo y solapamiento.
Se recorre el documento párrafo por párrafo (separados por línea en blanco) y
se van acumulando en un buffer hasta acercarse a `chunk_size_tokens`. Esto
preserva unidades semánticas completas (cada pregunta+respuesta del FAQ queda
junta) en vez de cortar a mitad de una oración, como pasaría con un chunking
puramente por caracteres. Cada chunk nuevo arranca repitiendo las últimas
`overlap_tokens` del chunk anterior para no perder contexto en los bordes.
"""

import tiktoken

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DATA_PATH,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    TOKENIZER_ENCODING,
)

_encoder = tiktoken.get_encoding(TOKENIZER_ENCODING)


# abro el archivo del documento y lo leo como texto; si falla con utf-8 (raro, pero por las dudas) reintento con latin-1
def load_document(path=DATA_PATH) -> str:
    """Lee el documento fuente como texto plano en UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# cuento cuántos tokens tiene un texto, usando el mismo tokenizer en todo el archivo
def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


# esto es para el caso raro de un párrafo gigante: lo corto oración por oración para que no se pase del máximo de tokens
def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """Divide un párrafo demasiado largo por oraciones para respetar max_tokens."""
    sentences = [s.strip() for s in paragraph.replace("\n", " ").split(". ") if s.strip()]
    pieces, buffer = [], ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}.".strip()
        if buffer and _count_tokens(candidate) > max_tokens:
            pieces.append(buffer)
            buffer = f"{sentence}."
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    return pieces


# acá está el corazón del chunking: voy juntando párrafos hasta acercarme al tamaño objetivo,
# y cuando cierro un chunk, el siguiente arranca repitiendo un poco del anterior (el solapamiento)
def chunk_document(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """Divide el texto en chunks de entre `min_tokens` y `max_tokens`.

    Devuelve una lista de dicts {chunk_id, text, num_tokens}.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    paragraphs = []
    for paragraph in raw_paragraphs:
        if _count_tokens(paragraph) > max_tokens:
            paragraphs.extend(_split_long_paragraph(paragraph, max_tokens))
        else:
            paragraphs.append(paragraph)

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip()
        if buffer and _count_tokens(candidate) > chunk_size_tokens:
            chunks.append(buffer)
            overlap_text = _tail_tokens(buffer, overlap_tokens)
            buffer = f"{overlap_text}\n\n{paragraph}".strip()
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)

    # Fusiona el último chunk con el anterior si quedó demasiado chico.
    if len(chunks) > 1 and _count_tokens(chunks[-1]) < min_tokens:
        last = chunks.pop()
        chunks[-1] = f"{chunks[-1]}\n\n{last}".strip()

    return [
        {"chunk_id": i, "text": chunk, "num_tokens": _count_tokens(chunk)}
        for i, chunk in enumerate(chunks)
    ]


# saco el final de un texto (los últimos n_tokens) para usarlo como solapamiento del próximo chunk
def _tail_tokens(text: str, n_tokens: int) -> str:
    """Devuelve los últimos n_tokens de un texto, decodificados de vuelta a string."""
    tokens = _encoder.encode(text)
    return _encoder.decode(tokens[-n_tokens:]) if n_tokens < len(tokens) else text
