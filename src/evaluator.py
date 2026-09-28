"""Agente evaluador: audita la calidad de una respuesta del sistema RAG.

Actúa como LLM-judge: recibe la pregunta, la respuesta generada y los chunks
usados como contexto, y devuelve un puntaje 0-10 junto con una justificación,
evaluando relevancia de los chunks, fidelidad de la respuesta al contexto
(para detectar alucinaciones) y completitud.
"""

import json
import re

from config import BASE_DIR
from llm_client import call_llm

EVALUATOR_PROMPT_PATH = BASE_DIR / "prompts" / "evaluator_prompt.txt"

_FALLBACK_EVALUATION = {
    "score": 0,
    "reason": "No se pudo evaluar la respuesta: el LLM evaluador no devolvió un JSON valido.",
}


# el LLM evaluador a veces mete texto extra alrededor del JSON, así que busco y parseo solo la parte que es JSON
def _extract_json(raw_text: str) -> dict:
    """Extrae el primer objeto JSON válido de la respuesta del LLM evaluador."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("La respuesta del evaluador no contiene un JSON")
    return json.loads(match.group(0))


# le paso la pregunta, la respuesta y los chunks a un LLM para que actúe de juez y me devuelva puntaje + justificación
def evaluate_response(
    user_question: str, system_answer: str, chunks_related: list[dict]
) -> dict:
    """Puntúa una respuesta del sistema RAG entre 0 y 10, con justificación."""
    chunks_text = "\n\n".join(
        f"[Chunk {c['chunk_id']}] {c['text']}" for c in chunks_related
    )
    template = EVALUATOR_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        question=user_question, answer=system_answer, chunks=chunks_text
    )

    response = call_llm(prompt)
    try:
        evaluation = _extract_json(response.text)
        return {
            "score": int(evaluation["score"]),
            "reason": str(evaluation["reason"]),
        }
    except (ValueError, KeyError, json.JSONDecodeError):
        return _FALLBACK_EVALUATION
