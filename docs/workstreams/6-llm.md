# Workstream 6: LLM behavior

The Oracle's brain: streaming chat with Ollama, persona prompt, conversation
memory with summarization, RAG-context wiring.

## Status

Working end-to-end. Llama 3.2 3B Q4_K_M via Ollama; SQLite-backed memory
with periodic LLM summarization; RAG context injected when collections
exist.

## Scope

- Async Ollama streaming client (`stream_chat`)
- Persona / system prompt builder from TOML
- Conversation memory (SQLite)
- Context window management (recent turns + summary)
- Conversation summarization (LLM call to compact older turns)
- RAG context plumbing (consumes Workstream 2's retriever)
- Greeting / sign-off lines

## File ownership

```
oracle/
  llm.py                   # async Ollama streaming client
  persona.py               # system prompt + greeting builder
  memory/
    __init__.py
    store.py               # SQLite conversation persistence
    context.py             # build LLM context (history + summary + RAG)
    summarizer.py          # compact older turns via the LLM itself
config/
  persona.toml             # personality, voice, behavior knobs
tests/
  test_llm.py
  test_persona.py
  test_memory.py
```

## Settings

```bash
ORACLE_OLLAMA_HOST=http://localhost:11434
ORACLE_OLLAMA_MODEL=llama3.2:3b
ORACLE_OLLAMA_TIMEOUT=120
ORACLE_DB_PATH=data/oracle.db
ORACLE_MAX_CONTEXT_TURNS=10
ORACLE_SUMMARY_THRESHOLD=20
```

`config/persona.toml` carries the prose: persona name, greeting, tone,
guardrails.

## Dependencies

```bash
pip install -e .                  # core httpx + pydantic-settings
# external: Ollama daemon
ollama pull llama3.2:3b
```

Optional `[rag]` for the RAG context path (lazy-imported).

## Interface contract

**Provides** (consumed by Workstreams 3, 4, 7):
- `async def stream_chat(messages: list[dict]) → AsyncIterator[str]`
- `async def check_ollama() → bool`
- `build_system_prompt() → str`
- `get_greeting() → str`
- `ConversationStore` — `new_session()`, `add_message()`, etc.
- `ContextBuilder.build(system_prompt, rag_context) → list[dict]`

**Consumes**:
- Workstream 2 (RAG) at runtime via `oracle.rag.retriever.Retriever` —
  lazy-imported, returns empty string if collections are missing

**Concurrency rule**: STT and LLM run **sequentially** (never concurrently)
to fit in the Jetson's 8 GB unified memory.

## Standalone exercise

```bash
# Confirm Ollama is reachable and the model is loaded
python -c "
import asyncio
from oracle.llm import check_ollama, stream_chat
print('ollama up?', asyncio.run(check_ollama()))
async def go():
    async for tok in stream_chat([
        {'role':'system','content':'You are concise.'},
        {'role':'user','content':'What is 2+2?'},
    ]):
        print(tok, end='', flush=True)
asyncio.run(go())
"

# Full text REPL — exercises persona + memory + RAG without TTS or hardware
python -m oracle --mode text

pytest tests/test_llm.py tests/test_persona.py tests/test_memory.py
```

## TODO

- [ ] Tool-use / function-calling for music + book intents (vs string matching)
- [ ] Multi-turn RAG (refine retrieval using the conversation context)
- [ ] Per-session token budget tracking
- [ ] Faster summarization model (smaller LLM dedicated to summaries)
- [ ] Conversation export (`/export` slash command in text REPL)
- [ ] Hot-reload `persona.toml` without restarting
