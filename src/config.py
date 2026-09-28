"""Configuración centralizada del proyecto: paths, parámetros de chunking,
modelo de embeddings y variables de entorno para el LLM."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "faq_document.txt"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

# --- Chunking (estrategia: por párrafos con tamaño objetivo y solapamiento) ---
# Se acumulan párrafos completos del documento hasta acercarse a CHUNK_SIZE_TOKENS,
# preservando límites de oración/párrafo en vez de cortar a mitad de una idea.
# Cada chunk final arranca repitiendo las últimas CHUNK_OVERLAP_TOKENS del chunk
# anterior, para no perder contexto en los bordes.
CHUNK_SIZE_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 480
TOKENIZER_ENCODING = "cl100k_base"

# --- Embeddings (Sentence-Transformers, local, sin API key) ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# --- Búsqueda vectorial ---
TOP_K = 4  # cantidad de chunks a recuperar por consulta (entre 2 y 5)

# Umbral mínimo de similitud (coseno) que tiene que tener el mejor chunk
# encontrado para considerar que la pregunta tiene relación con el documento.
# Se definió probando manualmente preguntas dentro y fuera de dominio: las
# relevantes (incluso parafraseadas) dieron scores >= 0.565, mientras que una
# pregunta claramente fuera de tema dio 0.494. 0.5 queda en el medio de ese
# rango con margen para ambos lados.
MIN_RELEVANCE_SCORE = 0.5

# --- LLM de generación (mismo patrón que el TP1: Gemini primario, OpenAI fallback) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
