"""FastAPI entrypoint — load `.env` from repo root before any settings-backed imports use env."""

from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of `app/` (works when cwd is not the project folder).
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import clear_settings_cache
from app.routes import api_router
from app.services.retrieval import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    clear_settings_cache()
    try:
        app.state.retriever = HybridRetriever.from_disk()
        app.state.retriever_error = None
        logger.info("Hybrid retriever loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retriever not initialized: %s", exc)
        app.state.retriever = None
        app.state.retriever_error = str(exc)
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "shl-assessment-recommender", "docs": "/docs"}
