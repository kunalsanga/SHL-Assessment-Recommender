# SHL Assessment Recommender (Take-Home)

Production-style FastAPI service that recommends **SHL Individual Test Solutions** only (pre-packaged Job Solutions are excluded at scrape time). Recommendations are **grounded in a local catalog** with **FAISS + hybrid lexical scoring**; the model only narrates over candidate rows chosen by code, and URLs are **allowlisted** to prevent hallucinated links.

## Features

- `GET /health` → `{"status":"ok"}`
- `POST /chat` → strict schema: `reply`, `recommendations`, `end_of_conversation` (no extra keys)
- **Hybrid retrieval**: MiniLM embeddings + keyword overlap
- **Structured state** extracted each turn (role, seniority, skills, preferences)
- **Comparison mode** grounded in catalog snippets only
- **Rule-based refusal** for obvious prompt-injection / scope issues
- **Max 8 user turns** per thread

## Repository layout

```text
app/
  main.py                 # FastAPI app + lifespan (loads retriever)
  config.py               # pydantic-settings
  routes/                 # /health, /chat
  services/               # retrieval, ranking, chat orchestration, Gemini client
  models/                 # pydantic schemas + catalog models
  prompts/                # prompt templates
  embeddings/             # sentence-transformers wrapper
  utils/                  # token overlap + scoring helpers
  data/                   # generated artifacts (catalog, FAISS, embeddings)
scripts/
  scrape_catalog.py       # BeautifulSoup + requests scraper
  build_embeddings.py
  build_faiss_index.py
Dockerfile
requirements.txt
```

## Prerequisites

- Python **3.11+**
- A **Google AI Studio / Gemini API key** with access to a **Flash** model (default: `gemini-2.5-flash`; override via `GEMINI_MODEL`, e.g. `gemini-1.5-flash` if needed)

## Local setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 1) Scrape the catalog (Individual Test Solutions only)

This hits SHL’s public catalog page and (by default) follows each product URL to enrich metadata.

```powershell
python scripts/scrape_catalog.py --out app/data/catalog.json
```

> Note: this can take a while (hundreds of detail pages). For a quick smoke run, use `--no-details`.

### 2) Build embeddings + FAISS

```powershell
python scripts/build_embeddings.py --catalog app/data/catalog.json --out-emb app/data/embeddings.npy --out-meta app/data/catalog_meta.json
python scripts/build_faiss_index.py --embeddings app/data/embeddings.npy --out app/data/faiss.index
```

### 3) Configure environment

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

### 4) Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

## Example requests

### Health

```powershell
curl http://localhost:8000/health
```

Response:

```json
{"status":"ok"}
```

### Chat (vague → clarifying; recommendations must be empty)

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I need an assessment\"}]}"
```

Example shape:

```json
{
  "reply": "…",
  "recommendations": [],
  "end_of_conversation": false
}
```

### Chat (specific hiring → 1–10 catalog-backed recommendations)

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I'm hiring a mid-level Java developer. I want technical knowledge tests plus a personality measure for team fit. Remote delivery is required.\"}]}"
```

### Chat (compare)

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What's the difference between OPQ and GSA?\"}]}"
```

## Testing / evaluation notes (interview-friendly)

- **Schema compliance**: `ChatResponse` uses Pydantic `extra="forbid"` so the API won’t silently emit unknown keys.
- **URL hallucination guard**: recommendations are assembled from the catalog in code and filtered with `allowlisted_urls()` before returning.
- **Retrieval quality**: tune weights in `app/services/recommendation.py` and `app/utils/scoring.py`; increase `retrieval_top_k` in `app/config.py`.
- **Latency**: first embedding call downloads MiniLM weights; subsequent requests are typically much faster. Keep `GEMINI_MODEL` on a Flash variant for sub-30s responses.

## Deployment (Render)

1. Create a **Web Service** from this repo.
2. **Build Command**:

   ```bash
   pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   ```

3. **Start Command**:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

4. Upload or generate `app/data/catalog.json`, `app/data/embeddings.npy`, `app/data/catalog_meta.json`, and `app/data/faiss.index` as part of your deploy strategy:
   - **Common pattern**: run the scrape + index scripts in CI, store artifacts in object storage or a release asset, and download them at container start (not implemented here to keep dependencies minimal).
   - **Simpler pattern for demos**: commit the generated JSON/index files if allowed by the assignment (check file size limits).

5. Set environment variables in Render:
   - `GEMINI_API_KEY` (**required** for `/chat`)
   - Optional: `GEMINI_MODEL`, `MAX_CONVERSATION_TURNS`, paths (`CATALOG_PATH`, etc.)

### Docker (local / any host)

```bash
docker build -t shl-recommender .
docker run --rm -p 8000:8000 --env-file .env shl-recommender
```

## Operational caveats

- SHL may change HTML; the scraper targets the **“Individual Test Solutions”** table header. If scraping fails, update selectors in `scripts/scrape_catalog.py`.
- The LLM is used for **state extraction** and **natural-language replies**; it is **not** the source of truth for URLs or which rows are returned.

## License / attribution

This project is a skills demonstration scaffold. SHL product names, URLs, and descriptions are owned by SHL; use scraped data responsibly and in line with SHL’s terms of use.
