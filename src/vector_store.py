"""Almacenamiento y búsqueda vectorial con FAISS.

Método: k-NN exacto (`IndexFlatIP`, producto interno) sobre embeddings
normalizados en L2. Producto interno de vectores normalizados equivale
matemáticamente a similitud coseno, así que esto nos da un ranking por
coseno explícito sin tener que reimplementar la fórmula a mano. Para el
tamaño de este dataset (decenas/cientos de chunks) un índice exacto es
más que suficiente en velocidad; no hace falta un índice aproximado (ANN).
"""

import json

import faiss
import numpy as np

from config import CHUNKS_PATH, EMBEDDING_DIMENSION, FAISS_INDEX_PATH, INDEX_DIR


# creo el índice de FAISS (vacío) y le cargo todos los embeddings de los chunks
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Crea un índice FAISS de producto interno (~coseno) a partir de embeddings."""
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)
    return index


# guardo el índice y el texto de los chunks en disco, así no tengo que recalcular todo cada vez que consulto
def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    """Persiste el índice FAISS y el texto de los chunks en disco."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


# cargo del disco el índice y los chunks que ya había guardado con save_index
def load_index() -> tuple[faiss.Index, list[dict]]:
    """Carga el índice FAISS y los chunks previamente guardados."""
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el índice. Corré primero: python src/build_index.py"
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


# busco en el índice los top_k chunks que más se parecen (coseno) al embedding de la pregunta
def search_similar_chunks(
    index: faiss.Index, chunks: list[dict], query_vector: np.ndarray, top_k: int
) -> list[dict]:
    """Busca los `top_k` chunks más similares (coseno) a un embedding de consulta."""
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
            }
        )
    return results
