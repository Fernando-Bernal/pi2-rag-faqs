import pytest
from pydantic import ValidationError

from schema import validate_query_result

VALID_RESULT = {
    "user_question": "¿Cómo solicito vacaciones?",
    "system_answer": "Desde la sección Mis Licencias, eligiendo el rango de fechas.",
    "chunks_related": [
        {"chunk_id": 0, "text": "texto del chunk", "score": 0.87},
    ],
}


def test_valid_result_passes():
    validate_query_result(VALID_RESULT)


def test_missing_key_fails():
    invalid = {k: v for k, v in VALID_RESULT.items() if k != "system_answer"}
    with pytest.raises(ValidationError):
        validate_query_result(invalid)


def test_wrong_type_fails():
    invalid = {**VALID_RESULT, "chunks_related": "no debería ser un string"}
    with pytest.raises(ValidationError):
        validate_query_result(invalid)


def test_result_has_exactly_the_three_required_keys():
    result = validate_query_result(VALID_RESULT)
    assert set(result.model_dump().keys()) == {
        "user_question",
        "system_answer",
        "chunks_related",
    }
