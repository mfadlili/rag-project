from fastapi import APIRouter, HTTPException

from app.rag import get_rag_service
from app.schemas import QuestionInput, RagResponse

router = APIRouter()


@router.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Selamat datang di Chatbot Panduan Operasional NusantaraCare"}


@router.post("/rag/", response_model=RagResponse)
def answer_with_rag(question: QuestionInput) -> RagResponse:
    try:
        service = get_rag_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503, detail="RAG service is not configured."
        )
    return service.answer(question.question)

