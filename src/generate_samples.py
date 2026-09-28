"""Corre el pipeline de consulta end-to-end sobre preguntas de ejemplo y
guarda los resultados en outputs/, para demostrar el sistema funcionando
de punta a punta (y, opcionalmente, el agente evaluador sobre esos mismos
resultados).

Uso:
    python src/generate_samples.py
"""

import json

from config import OUTPUTS_DIR
from evaluator import evaluate_response
from query import run_query

SAMPLE_QUESTIONS = [
    "¿Cómo solicito una licencia por vacaciones?",  # pregunta de "cómo hacer"
    "¿NexHR tiene certificación de seguridad SOC 2?",  # pregunta de sí/no
    "¿Qué pasa si un empleado tiene un error en su pago?",
    "¿Cuáles son los límites de uso de la API?",
    "¿Qué beneficios se pueden gestionar desde la plataforma?",
]


# corro todas las preguntas de ejemplo contra el pipeline completo y guardo las respuestas y evaluaciones en outputs/
def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    sample_queries = []
    sample_evaluations = []

    for question in SAMPLE_QUESTIONS:
        print(f"-> Consultando: {question}")
        result = run_query(question)
        sample_queries.append(result)

        evaluation = evaluate_response(
            result["user_question"], result["system_answer"], result["chunks_related"]
        )
        sample_evaluations.append({"user_question": question, **evaluation})

    queries_path = OUTPUTS_DIR / "sample_queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(sample_queries, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {queries_path}")

    evaluations_path = OUTPUTS_DIR / "sample_evaluations.json"
    with open(evaluations_path, "w", encoding="utf-8") as f:
        json.dump(sample_evaluations, f, ensure_ascii=False, indent=2)
    print(f"Guardado: {evaluations_path}")


if __name__ == "__main__":
    main()
