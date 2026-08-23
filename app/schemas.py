from pydantic import BaseModel, Field
from typing import Literal



class QuestionInput(BaseModel):
    question: str = Field(min_length=1)


class SourceModel(BaseModel):
    source: str
    document_version: str

    policy_version: str
    policy_status: str
    is_active: bool

    section: str | None = None
    subsection: str | None = None

    chunk: int
    chunk_id: str

    text: str
    distance: float


class RagResponse(BaseModel):
    answer: str
    confidence_label : Literal["high", "medium", "low"]
    reason_code: Literal[
        "answered",               # Jawaban berhasil dijawab dari sumber
        "no_relevant_context",    # Dokumen rujukan tidak memiliki jawaban (Out-of-Scope)
        "conflicting_sources",    # Dokumen rujukan saling bertentangan secara isi
        "unauthorized_access"     # Pengguna tidak memiliki akses ke dokumen tersebut
    ]
    sources: list[SourceModel]