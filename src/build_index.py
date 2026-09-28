"""Pipeline de indexación (Data Pipeline):

1. Carga del documento fuente.
2. Segmentación en chunks (chunking por párrafos con solapamiento).
3. Generación de embeddings (Sentence-Transformers).
4. Almacenamiento de embeddings + texto de cada chunk (FAISS + JSON).

Uso:
    python src/build_index.py
"""

from chunking import chunk_document, load_document
from config import DATA_PATH
from embeddings import generate_embeddings
from vector_store import build_faiss_index, save_index


# esta es la que corro para armar todo de cero: carga el documento, lo chunkea, genera los embeddings y guarda el índice
def main() -> None:
    print(f"[1/4] Cargando documento: {DATA_PATH}")
    text = load_document()

    print("[2/4] Segmentando en chunks...")
    chunks = chunk_document(text)
    print(f"      -> {len(chunks)} chunks generados")
    token_counts = [c["num_tokens"] for c in chunks]
    print(f"      -> tokens por chunk: min={min(token_counts)}, max={max(token_counts)}")

    print("[3/4] Generando embeddings (Sentence-Transformers)...")
    embeddings = generate_embeddings(chunks)
    print(f"      -> {embeddings.shape[0]} embeddings de dimensión {embeddings.shape[1]}")

    print("[4/4] Guardando índice FAISS + chunks en disco...")
    index = build_faiss_index(embeddings)
    save_index(index, chunks)
    print("      -> índice guardado en index/faiss.index e index/chunks.json")

    print("\nIndexación completa.")


if __name__ == "__main__":
    main()
