# TP2 - Chatbot de Soporte para FAQs con RAG

Chatbot de soporte al cliente basado en **RAG (Retrieval-Augmented Generation)** para NexHR, una plataforma ficticia de RRHH. En lugar de que un agente humano busque manualmente en la documentación interna, el sistema recupera automáticamente los fragmentos más relevantes de un documento de FAQs y usa un LLM para generar una respuesta precisa y fundamentada en esos fragmentos, devolviendo siempre un JSON estructurado (`user_question`, `system_answer`, `chunks_related`) que permite auditar en qué información se basó cada respuesta.

Proyecto integrador — Módulo 2, AI Engineering (Henry).

## ¿Por qué RAG?

Este sistema usa RAG porque **primero recupera** los chunks de documentación más relevantes para la pregunta (búsqueda vectorial) y **recién después genera** la respuesta, pasándole esos chunks al LLM como contexto (ver `src/query.py`, funciones `search_similar_chunks` → `build_context` → `generate_answer`). Esto evita tener que reentrenar o hacer fine-tuning del modelo cada vez que cambia la documentación: si el documento de FAQs se actualiza, el sistema responde con la información nueva apenas se vuelve a correr la indexación. Además, como cada respuesta viene acompañada de los chunks que la sustentan, el proceso queda **auditable y con atribución de fuente** (se puede verificar exactamente de dónde salió cada dato).

## Arquitectura (resumen)

```
data/faq_document.txt
      -> build_index.py    (carga + chunking + embeddings + guardado en FAISS)
      -> index/faiss.index + index/chunks.json

pregunta del usuario
      -> query.py           (embedding de la pregunta)
      -> vector_store.py    (búsqueda k-NN por similitud coseno en FAISS)
      -> query.py           (¿el mejor chunk supera MIN_RELEVANCE_SCORE?)
            NO -> respuesta fija "no tengo información" (no se llama al LLM)
            SI -> arma el contexto con los chunks recuperados
                  -> llm_client.py (Gemini primario, fallback OpenAI, genera la respuesta)
      -> schema.py          (valida el JSON contra el contrato, con Pydantic)
      -> JSON final: user_question, system_answer, chunks_related

evaluator.py (opcional, sobre el JSON anterior)
      -> LLM-judge que puntúa la respuesta 0-10 con justificación
```

## Requisitos

- Python 3.10+
- Una API key de [Google AI Studio](https://aistudio.google.com/) (Gemini, proveedor primario de generación) y/o de [OpenAI](https://platform.openai.com/) (fallback)
- No se necesita API key para los embeddings: se generan localmente con Sentence-Transformers.

## Instalación y configuración

1. Cloná el repositorio y creá un entorno virtual:

   ```powershell
   git clone <url-de-este-repo>
   cd tp2-rag-faqs

   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Instalá las dependencias:

   ```powershell
   pip install -r requirements.txt
   ```

3. Configurá tu API key copiando la plantilla de variables de entorno y completando tus claves reales:

   ```powershell
   Copy-Item .env.example .env
   # export OPENAI_API_KEY=tu-key-aqui   (o completar GEMINI_API_KEY en el .env)
   ```

`.env` nunca se sube al repositorio (está en `.gitignore`).

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | Proveedor primario de generación: `gemini` u `openai` (default: `gemini`) |
| `GEMINI_API_KEY` | API key de Google AI Studio |
| `OPENAI_API_KEY` | API key de OpenAI (usada como fallback si Gemini falla) |
| `GEMINI_MODEL` | Modelo de Gemini a usar (default: `gemini-2.5-flash-lite`) |
| `OPENAI_MODEL` | Modelo de OpenAI a usar (default: `gpt-4o-mini`) |
| `EMBEDDING_MODEL` | Modelo de Sentence-Transformers para embeddings (default: `all-MiniLM-L6-v2`) |

## Cómo ejecutarlo

### 1. Pipeline de indexación

Procesa `data/faq_document.txt`, lo divide en chunks, genera embeddings y guarda el índice:

```powershell
python src/build_index.py
```

Esto crea `index/faiss.index` (vectores) e `index/chunks.json` (texto + metadata de cada chunk).

### 2. Pipeline de consultas

Con el índice ya generado, se puede preguntar directamente por consola:

```powershell
python src/query.py "¿Cómo solicito una licencia por vacaciones?"
```

Salida esperada (formato JSON, siempre con estas tres claves):

```json
{
  "user_question": "¿Cómo solicito una licencia por vacaciones?",
  "system_answer": "Puedes solicitar vacaciones desde la sección \"Mis Licencias\"...",
  "chunks_related": [
    {"chunk_id": 8, "text": "...", "score": 0.69},
    {"chunk_id": 10, "text": "...", "score": 0.66}
  ]
}
```

### 3. Generar ejemplos y evaluarlos (bonus)

```powershell
python src/generate_samples.py
```

Corre 5 preguntas de ejemplo (incluyendo casos de sí/no y de "cómo hacer") de punta a punta y guarda:
- `outputs/sample_queries.json`: los JSON de respuesta del pipeline de consultas.
- `outputs/sample_evaluations.json`: el puntaje (0-10) y la justificación del agente evaluador para cada respuesta.

### 4. Tests

```powershell
pytest tests/ -v
```

Los tests validan la segmentación en chunks (cantidad y límites de tokens), el contrato JSON (casos válidos e inválidos) y el parseo robusto del JSON del agente evaluador. No requieren API key ni hacen llamadas reales.

## Estructura del proyecto

```
data/faq_document.txt       Documento fuente de FAQs (~1900 palabras, 8 secciones temáticas)
prompts/                    Templates de prompt para generación y evaluación
src/
  config.py                 Paths, parámetros de chunking y configuración de proveedores
  chunking.py                load_document(), chunk_document()
  embeddings.py              generate_embeddings(), embed_query()
  vector_store.py            build_faiss_index(), save_index(), load_index(), search_similar_chunks()
  llm_client.py               call_llm() con fallback Gemini -> OpenAI
  schema.py                   validate_query_result() (contrato JSON con Pydantic)
  build_index.py              Orquesta el pipeline de indexación (main)
  query.py                    Orquesta el pipeline de consulta (main); has_relevant_chunks() aplica el umbral de relevancia
  evaluator.py                 evaluate_response() (agente evaluador LLM-judge)
  generate_samples.py         Genera outputs/sample_queries.json y sample_evaluations.json
index/                       Índice FAISS + chunks generados (no se versiona, se regenera con build_index.py)
outputs/sample_queries.json  Ejemplos de consulta-respuesta de punta a punta
tests/                       Tests unitarios (chunking, schema, evaluator)
```

## Decisiones técnicas

**Chunking**: se eligió *chunking por párrafos con tamaño objetivo y solapamiento* (`src/chunking.py`). El documento se recorre párrafo por párrafo (cada pregunta+respuesta de la FAQ es un párrafo natural) y se van acumulando hasta acercarse a `CHUNK_SIZE_TOKENS=220`, en vez de cortar el texto cada N caracteres. Esto preserva unidades semánticas completas (nunca se corta una respuesta a la mitad) y cada chunk nuevo repite las últimas `CHUNK_OVERLAP_TOKENS=40` del anterior para no perder contexto en los bordes. Los tokens se cuentan con `tiktoken` para garantizar que cada chunk quede entre `MIN_CHUNK_TOKENS=50` y `MAX_CHUNK_TOKENS=480`.

**Búsqueda vectorial**: se usa **FAISS `IndexFlatIP`** (k-NN exacto por producto interno) sobre embeddings normalizados en L2, lo que equivale matemáticamente a **similitud coseno**. Para el tamaño de este dataset (decenas de chunks) un índice exacto es rápido de sobra; no hace falta un índice aproximado (ANN), pero FAISS permite escalar a ANN (`IndexIVFFlat`, `IndexHNSWFlat`) sin cambiar el resto del pipeline si la base de conocimiento creciera mucho.

**Embeddings**: se usa `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones), un modelo local liviano que no requiere API key ni tiene costo por token, ideal para indexar documentación interna con frecuencia sin generar gasto recurrente.

**LLM de generación y evaluación**: se reutiliza el mismo patrón del proyecto del Módulo 1 (Gemini como proveedor primario por costo, con fallback automático a OpenAI si Gemini falla), configurable con `LLM_PROVIDER` en `.env`.

**Umbral de relevancia mínima (`MIN_RELEVANCE_SCORE=0.5` en `config.py`)**: antes de llamar al LLM, `has_relevant_chunks()` (`src/query.py`) chequea si el chunk mejor rankeado supera este score de similitud coseno. Si ninguno lo supera, se devuelve directamente una respuesta de "no tengo información sobre eso" **sin gastar tokens en el LLM**, en vez de arriesgarse a que el modelo alucine una respuesta con contexto irrelevante. El valor 0.5 se definió probando manualmente preguntas dentro y fuera de dominio: preguntas relevantes (incluso parafraseadas, como "¿con que beneficios cuento?") dieron scores ≥ 0.565, mientras que una pregunta claramente fuera de tema ("¿puedo jugar al pádel en horario laboral?") dio 0.494. Con una sola muestra de cada lado, 0.5 es un punto de partida razonable con margen a ambos lados, no un valor definitivo — con más casos de prueba podría ajustarse.

## Limitaciones conocidas

- La búsqueda de chunks es puramente semántica (embeddings); no se implementó un componente de búsqueda por palabra clave (BM25) para una estrategia híbrida, aunque el código de `vector_store.py` está aislado como para agregarlo sin tocar el resto del pipeline.
- `EMBEDDING_DIMENSION` está fijo en 384 (dimensión de `all-MiniLM-L6-v2`); si se cambia `EMBEDDING_MODEL` por uno de otra dimensionalidad, hay que actualizar esa constante en `config.py`.
- El agente evaluador es un LLM-judge (no una métrica determinística tipo ROUGE/BLEU), por lo que su puntaje puede variar levemente entre corridas para la misma pregunta.
- Si el LLM de generación devuelve algo que no matchea el contrato JSON esperado, `schema.py` lanza una excepción de validación en vez de reintentar automáticamente la llamada.
- El plan gratuito de Gemini tiene un límite bajo de requests diarios (20/día para `gemini-2.5-flash-lite` al momento de escribir esto). Si se supera, `llm_client.py` cae automáticamente a OpenAI, así que el sistema sigue funcionando, pero conviene tenerlo en cuenta al probar muchas preguntas seguidas.
- `requirements.txt` fija `httpx==0.27.2` porque `openai==1.35.13` es incompatible con `httpx>=0.28` (versiones nuevas de `httpx` sacaron el parámetro `proxies` que esa versión de la librería de OpenAI todavía usa internamente).
