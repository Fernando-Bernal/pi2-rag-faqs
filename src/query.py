"""Pipeline de consulta (Query Pipeline):

1. Embedding de la pregunta del usuario.
2. Búsqueda vectorial (k-NN sobre FAISS) de los chunks más relevantes.
3. Ensamblado del contexto recuperado en el prompt.
4. Generación de la respuesta con un LLM (RAG: primero se recupera, después
   se genera; esto le da al LLM conocimiento específico del dominio sin
   necesidad de fine-tuning, y permite auditar en qué chunks se basó la
   respuesta).

Si ningún chunk recuperado supera MIN_RELEVANCE_SCORE (pregunta fuera de
dominio), se devuelve una respuesta fija sin llamar al LLM: evita alucinar
una respuesta inventada y ahorra tokens en preguntas que el documento no
puede responder.

Uso:
    python src/query.py "¿Cómo solicito una licencia por vacaciones?"
"""

import json
import sys

from config import BASE_DIR, MIN_RELEVANCE_SCORE, TOP_K
from embeddings import embed_query
from llm_client import call_llm
from schema import validate_query_result
from vector_store import load_index, search_similar_chunks

PROMPT_TEMPLATE_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"

NO_RELEVANT_INFO_MESSAGE = (
    "No encontré información relevante en la documentación de FAQs para "
    "responder esta pregunta. Te recomendamos consultar con un agente de soporte."
)


# junto el texto de los chunks recuperados en un solo bloque, numerado, para pasárselo al LLM como contexto
def build_context(chunks_related: list[dict]) -> str:
    """Formatea los chunks recuperados como contexto numerado para el prompt."""
    return "\n\n".join(
        f"[Chunk {c['chunk_id']}] {c['text']}" for c in chunks_related
    )


# tomo el template del prompt, le meto la pregunta y el contexto, y le pido al LLM que responda
def generate_answer(question: str, context: str) -> str:
    """Genera la respuesta final llamando al LLM con la pregunta + contexto."""
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt = template.format(context=context, question=question)
    response = call_llm(prompt)
    return response.text.strip()


# si el mejor chunk encontrado no llega al umbral mínimo, la pregunta probablemente está fuera de dominio: no vale la pena (ni es seguro) llamar al LLM
def has_relevant_chunks(
    chunks_related: list[dict], min_score: float = MIN_RELEVANCE_SCORE
) -> bool:
    """True si el chunk mejor rankeado supera el umbral mínimo de relevancia."""
    return bool(chunks_related) and chunks_related[0]["score"] >= min_score


# esta es la que conecta todo el pipeline de consulta: cargo el índice, busco los chunks, genero la respuesta (o corto antes si no hay nada relevante) y valido el JSON final
def run_query(question: str, top_k: int = TOP_K) -> dict:
    """Orquesta el pipeline completo de consulta y devuelve el JSON de salida."""
    index, chunks = load_index()

    query_vector = embed_query(question)
    chunks_related = search_similar_chunks(index, chunks, query_vector, top_k)

    if has_relevant_chunks(chunks_related):
        context = build_context(chunks_related)
        system_answer = generate_answer(question, context)
    else:
        system_answer = NO_RELEVANT_INFO_MESSAGE

    result = {
        "user_question": question,
        "system_answer": system_answer,
        "chunks_related": chunks_related,
    }
    validate_query_result(result)  # lanza ValidationError si el contrato no se cumple
    return result


# punto de entrada por consola: agarro la pregunta que le paso como argumento y muestro el JSON de respuesta
def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python src/query.py "tu pregunta"')
        sys.exit(1)

    question = sys.argv[1]
    result = run_query(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
