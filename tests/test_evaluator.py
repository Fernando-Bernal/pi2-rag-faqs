from evaluator import _extract_json


def test_extract_json_parses_clean_json():
    raw = '{"score": 8, "reason": "Responde usando los chunks relevantes."}'
    result = _extract_json(raw)
    assert result == {"score": 8, "reason": "Responde usando los chunks relevantes."}


def test_extract_json_ignores_surrounding_text():
    raw = 'Claro, aca esta mi evaluacion:\n{"score": 5, "reason": "Le falta detalle."}\nEspero que sirva.'
    result = _extract_json(raw)
    assert result["score"] == 5


def test_extract_json_raises_without_json():
    try:
        _extract_json("esto no tiene json")
        assert False, "debería haber lanzado ValueError"
    except ValueError:
        pass
