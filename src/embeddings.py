"""Generación de embeddings con Sentence-Transformers (local, sin API key)."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

_model = None


# cargo el modelo de sentence-transformers una sola vez (es pesado) y lo reutilizo en cada llamada
def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


# transformo todos los chunks de texto en vectores (embeddings), normalizados para poder comparar por coseno después
def generate_embeddings(chunks: list[dict]) -> np.ndarray:
    """Genera un embedding normalizado (L2) por cada chunk de texto.

    La normalización permite usar producto punto en FAISS como equivalente
    exacto de similitud coseno.
    """
    texts = [chunk["text"] for chunk in chunks]
    model = _get_model()
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(embeddings, dtype="float32")


# hago lo mismo pero con la pregunta del usuario, para poder compararla contra los embeddings de los chunks
def embed_query(question: str) -> np.ndarray:
    """Convierte una pregunta de usuario en un embedding normalizado."""
    model = _get_model()
    embedding = model.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(embedding, dtype="float32")
