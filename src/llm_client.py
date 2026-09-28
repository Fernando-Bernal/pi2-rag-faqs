"""Capa de abstracción sobre proveedores de LLM para la generación de respuestas.

Mismo criterio que en el proyecto del Módulo 1: Gemini es el proveedor
primario (por costo) y OpenAI actúa como fallback si Gemini falla.
"""

from dataclasses import dataclass

from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


# llamo a la API de Gemini con el prompt ya armado y devuelvo el texto que respondió
def _call_gemini(prompt: str) -> LLMResponse:
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY no configurada en .env")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt, generation_config={"temperature": 0.2})
    return LLMResponse(text=response.text, provider="gemini", model=GEMINI_MODEL)


# lo mismo pero con OpenAI; esta la uso como fallback si Gemini falla
def _call_openai(prompt: str) -> LLMResponse:
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return LLMResponse(
        text=response.choices[0].message.content, provider="openai", model=OPENAI_MODEL
    )


# según lo que configuré en LLM_PROVIDER, uso Gemini u OpenAI; si Gemini falla, reintento con OpenAI automáticamente
def call_llm(prompt: str) -> LLMResponse:
    """Llama al proveedor primario configurado en LLM_PROVIDER; si es 'gemini'
    y falla, reintenta automáticamente con OpenAI."""
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt)

    try:
        return _call_gemini(prompt)
    except Exception as gemini_error:
        print(f"[llm_client] Gemini fallo ({gemini_error}); usando fallback OpenAI...")
        return _call_openai(prompt)
