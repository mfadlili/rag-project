from functools import lru_cache

import chromadb
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
load_dotenv()

from app.config import (
    CHAT_MODEL,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    get_openai_client,
)
from app.schemas import RagResponse, SourceModel
from app.ingest import main as ingest_data



# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "nusantaracare_documents"

TOP_K = 5

NO_CONTEXT_RESPONSE = (
    "Informasi tidak ditemukan dalam dokumen."
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_MESSAGE = """
Anda adalah chatbot Panduan Operasional NusantaraCare.

Jawab pertanyaan HANYA berdasarkan CONTEXT yang diberikan.

Jangan menggunakan pengetahuan dari luar CONTEXT.

ATURAN JAWABAN:

1. Jika CONTEXT memiliki informasi yang secara langsung menjawab pertanyaan:
   - Berikan jawaban berdasarkan CONTEXT.
   - confidence_label = "high"
   - reason_code = "answered"

2. Jika CONTEXT relevan tetapi tidak memberikan jawaban yang lengkap:
   - Berikan jawaban hanya berdasarkan informasi yang tersedia.
   - confidence_label = "medium"
   - reason_code = "answered"

3. Jika CONTEXT tidak memiliki informasi yang relevan:
   - Jangan mengarang jawaban.
   - Jawab bahwa informasi tidak ditemukan di dokumen.
   - confidence_label = "low"
   - reason_code = "no_relevant_context"

4. Jika terdapat beberapa sumber yang memberikan ketentuan
   yang bertentangan:
   - Jangan memilih salah satu secara sembarangan.
   - Jelaskan bahwa terdapat konflik informasi.
   - confidence_label = "low"
   - reason_code = "conflicting_sources"

5. Jika CONTEXT menunjukkan bahwa informasi atau dokumen
   hanya dapat digunakan oleh pengguna yang memiliki akses tertentu:
   - Jika pengguna tidak memiliki akses, gunakan:
     reason_code = "unauthorized_access"
   - confidence_label = "low"

6. Dokumen dengan policy_status "inactive" atau is_active = false
   adalah dokumen historis dan TIDAK boleh digunakan sebagai
   ketentuan operasional saat ini.

7. Jika terdapat dokumen aktif dan dokumen tidak aktif yang
   memberikan ketentuan berbeda, prioritaskan dokumen aktif.

8. Untuk pertanyaan tentang kebijakan yang berlaku saat ini,
   jangan menggunakan kebijakan historis sebagai dasar jawaban.

OUTPUT:

Kembalikan JSON dengan format:

{
    "answer": "...",
    "confidence_label": "high | medium | low",
    "reason_code": "answered | no_relevant_context | conflicting_sources | unauthorized_access"
}
""".strip()


# ============================================================
# RAG SERVICE
# ============================================================

class RagService:

    def __init__(
        self,
        client: OpenAI,
    ):
        self.client = client

        # Connect to existing ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

         # Open existing collection
        self.collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    # ========================================================
    # EMBED USER QUESTION
    # ========================================================

    def embed_question(
        self,
        question: str,
    ):

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[question],
        )

        return response.data[0].embedding


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
    ) -> dict:

        query_embedding = self.embed_question(
            question
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return results


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def answer(
        self,
        question: str,
        top_k: int = TOP_K,
    ) -> RagResponse:

        results = self.retrieve(
            question,
            top_k=top_k,
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]


        # ----------------------------------------------------
        # No results
        # ----------------------------------------------------

        if not documents:

            return RagResponse(
                answer=NO_CONTEXT_RESPONSE,
                confidence_label="low",
                reason_code="no_relevant_context",
                sources=[],
            )


        # ----------------------------------------------------
        # Build AI context
        # ----------------------------------------------------

        context_parts = []

        sources = []


        for index, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            source_label = (
                f"Source {index}"
            )


            # -----------------------------------------------
            # Context for LLM
            # -----------------------------------------------

            context_parts.append(
                f"""
[{source_label}]

Document:
{metadata.get("doc_title", "")}

Document Version:
{metadata.get("document_version", "")}

Policy Version:
{metadata.get("policy_version", "")}

Policy Status:
{metadata.get("policy_status", "")}

Active:
{metadata.get("is_active", False)}

Effective Date:
{metadata.get("effective_date", "")}

Effective Until:
{metadata.get("effective_until", "")}

Section:
{metadata.get("section", "")}

Subsection:
{metadata.get("subsection", "")}

Chunk:
{metadata.get("chunk_number", "")}

Content:
{document}
""".strip()
            )


            # -----------------------------------------------
            # API citation
            # -----------------------------------------------

            sources.append(
                SourceModel(
                    source=source_label,

                    category=metadata.get(
                        "category",
                        "operational",
                    ),

                    chunk=int(
                        metadata.get(
                            "chunk_number",
                            0,
                        )
                    ),

                    chunk_id=metadata.get(
                        "chunk_id",
                        "",
                    ),

                    text=document,

                    document_version=metadata.get(
                        "document_version",
                        "",
                    ),

                    policy_version=metadata.get(
                        "policy_version",
                        "",
                    ),

                    policy_status=metadata.get(
                        "policy_status",
                        "",
                    ),

                    is_active=bool(
                        metadata.get(
                            "is_active",
                            False,
                        )
                    ),

                    section=metadata.get(
                        "section",
                        None,
                    ),

                    subsection=metadata.get(
                        "subsection",
                        None,
                    ),

                    distance=float(
                        distance
                    ),
                )
            )


        context = "\n\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # LLM prompt
        # ----------------------------------------------------

        user_message = f"""
CONTEXT:

{context}


PERTANYAAN:

{question}
""".strip()


                # ----------------------------------------------------
        # Call OpenAI
        # ----------------------------------------------------

        response = (
            self.client
            .chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE,
                    },
                    {
                        "role": "user",
                        "content": user_message,
                    },
                ],
                response_format={"type": "json_object"},  # enforce JSON if the endpoint supports it
            )
        )

        # ----------------------------------------------------
        # Get structured response
        # ----------------------------------------------------

        ai_answer = response.choices[0].message.content

        # Check BEFORE parsing, not after
        if not ai_answer:
            return RagResponse(
                answer=NO_CONTEXT_RESPONSE,
                confidence_label="low",
                reason_code="no_relevant_context",
                sources=sources,
            )

        # Strip markdown code fences if the model wraps its JSON
        cleaned = ai_answer.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            ai_result = json.loads(cleaned)
        except json.JSONDecodeError:
            return RagResponse(
                answer=NO_CONTEXT_RESPONSE,
                confidence_label="low",
                reason_code="no_relevant_context",
                sources=sources,
            )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return RagResponse(
            answer=ai_result.get("answer", NO_CONTEXT_RESPONSE),
            confidence_label=ai_result.get("confidence_label", "low"),
            reason_code=ai_result.get("reason_code", "no_relevant_context"),
            sources=sources,
        )

# ============================================================
# SINGLE SERVICE INSTANCE
# ============================================================

@lru_cache(maxsize=1)
def get_rag_service() -> RagService:

    client = get_openai_client()

    return RagService(client)