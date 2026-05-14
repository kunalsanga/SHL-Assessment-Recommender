from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import process_chat
from app.services.retrieval import HybridRetriever

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    retriever: HybridRetriever | None = getattr(request.app.state, "retriever", None)
    return process_chat(payload, retriever)
