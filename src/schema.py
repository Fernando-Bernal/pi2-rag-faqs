"""Contrato de salida del pipeline de consulta, validado con Pydantic."""

from pydantic import BaseModel, Field


class ChunkRelated(BaseModel):
    chunk_id: int
    text: str
    score: float


class QueryResult(BaseModel):
    user_question: str
    system_answer: str
    chunks_related: list[ChunkRelated] = Field(min_length=1)


# reviso que el diccionario que armé tenga exactamente las claves y tipos que definí en el contrato, si no, explota acá y no más adelante
def validate_query_result(result: dict) -> QueryResult:
    """Valida que un dict tenga exactamente el contrato esperado.

    Lanza pydantic.ValidationError si falta una clave o el tipo no coincide.
    """
    return QueryResult(**result)
