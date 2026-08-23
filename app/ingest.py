import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    get_openai_client,
)

load_dotenv()

COLLECTION_NAME = "nusantaracare_documents"

DOCUMENT_PATH = (
    "./documents/"
    "nusantaracare_panduan_operasional_internal_v2.md"
)

MAX_CHARS = 1500

OVERLAP_PARAGRAPHS = 1

# Batch size for embedding API calls (avoid sending too many texts in one request)
EMBED_BATCH_SIZE = 64


# ============================================================
# OpenAI-compatible client (used for embeddings now, not just chat)
# ============================================================

client = get_openai_client()

print(f"Using embedding model: {EMBEDDING_MODEL}")


# ============================================================
# Connect to ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = (
    chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )
)


# ============================================================
# Read Markdown document
# ============================================================

def read_document(path: str) -> str:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


# ============================================================
# Parse front matter
# ============================================================

def parse_front_matter(
    text: str
):

    metadata = {}

    lines = text.splitlines()

    if (
        not lines
        or lines[0].strip() != "---"
    ):
        return metadata, lines

    end_index = None

    for i in range(
        1,
        len(lines)
    ):

        if lines[i].strip() == "---":

            end_index = i
            break

    if end_index is None:
        return metadata, lines

    for line in lines[
        1:end_index
    ]:

        match = re.match(
            r"^([^:]+):\s*(.*)$",
            line
        )

        if not match:
            continue

        key = match.group(1).strip()
        value = match.group(2).strip()

        # Remove surrounding quotes
        value = (
            value
            .strip('"')
            .strip("'")
        )

        if value.lower() == "true":

            value = True

        elif value.lower() == "false":

            value = False

        metadata[key] = value

    return (
        metadata,
        lines[end_index + 1:]
    )


# ============================================================
# Detect policy metadata
# ============================================================

def detect_policy_metadata(
    section: str,
    subsection: str,
    document_metadata: dict,
):

    section_lower = section.lower()
    subsection_lower = subsection.lower()

    # --------------------------------------------------------
    # Historical v1.4
    # --------------------------------------------------------

    if (
        "arsip kebijakan v1.4"
        in section_lower

        or

        "arsip kebijakan v1.4"
        in subsection_lower
    ):

        return {
            "policy_version": "1.4",
            "policy_status": "archived",
            "is_active": False,
            "effective_date": "2025-01-01",
            "effective_until": "2026-06-30",
        }

    # --------------------------------------------------------
    # Explicit active v2.0 section
    # --------------------------------------------------------

    if (
        "pengganti aktif v2.0"
        in section_lower

        or

        "pengganti aktif v2.0"
        in subsection_lower
    ):

        return {
            "policy_version": "2.0",
            "policy_status": "active",
            "is_active": True,
            "effective_date": "2026-07-01",
            "effective_until": None,
        }

    # --------------------------------------------------------
    # Normal document content
    # --------------------------------------------------------

    return {
        "policy_version": str(
            document_metadata.get(
                "doc_version",
                "2.0",
            )
        ),

        "policy_status": "active",

        "is_active": True,

        "effective_date": (
            document_metadata.get(
                "effective_date"
            )
        ),

        "effective_until": None,
    }


# ============================================================
# Parse Markdown sections
# ============================================================

def parse_sections(
    lines: list[str]
):

    sections = []

    current_h2 = None
    current_h3 = None

    current_content = []

    content_start_line = None

    def save_content():

        nonlocal current_content
        nonlocal content_start_line

        if not current_content:
            return

        text = "\n".join(
            current_content
        ).strip()

        if not text:

            current_content = []
            content_start_line = None

            return

        sections.append(
            {
                "section": current_h2,
                "subsection": current_h3,
                "text": text,
                "line_start": content_start_line,
            }
        )

        current_content = []
        content_start_line = None


    for index, line in enumerate(
        lines
    ):

        line_number = index + 1

        # ----------------------------------------------------
        # H2
        # ----------------------------------------------------

        if re.match(
            r"^##\s+",
            line
        ):

            save_content()

            current_h2 = re.sub(
                r"^##\s+",
                "",
                line
            ).strip()

            current_h3 = None

            continue


        # ----------------------------------------------------
        # H3
        # ----------------------------------------------------

        if re.match(
            r"^###\s+",
            line
        ):

            save_content()

            current_h3 = re.sub(
                r"^###\s+",
                "",
                line
            ).strip()

            continue


        # ----------------------------------------------------
        # Ignore H1
        # ----------------------------------------------------

        if re.match(
            r"^#\s+",
            line
        ):

            save_content()

            continue


        # ----------------------------------------------------
        # Blank line
        # ----------------------------------------------------

        if not line.strip():

            if current_content:
                current_content.append("")

            continue


        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        if content_start_line is None:

            content_start_line = (
                line_number
            )

        current_content.append(
            line
        )


    save_content()

    return sections


# ============================================================
# Split sections into chunks
# ============================================================

def split_into_chunks(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_paragraphs: int = OVERLAP_PARAGRAPHS,
) -> list[str]:

    if not text.strip():
        return []

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be greater than 0"
        )

    if overlap_paragraphs < 0:
        raise ValueError(
            "overlap_paragraphs must be >= 0"
        )

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    current_chunk = ""

    current_paragraphs = []


    for paragraph in paragraphs:

        # ====================================================
        # Paragraph itself is larger than max_chars
        # ====================================================

        if len(paragraph) > max_chars:

            if current_chunk:

                chunks.append(
                    current_chunk
                )

            current_chunk = ""

            current_paragraphs = []

            words = paragraph.split()

            word_chunk = ""

            for word in words:

                candidate = (
                    word_chunk
                    + " "
                    + word
                ).strip()

                if len(candidate) <= max_chars:

                    word_chunk = candidate

                else:

                    if word_chunk:

                        chunks.append(
                            word_chunk
                        )

                    word_chunk = word

            if word_chunk:

                current_chunk = (
                    word_chunk
                )

                current_paragraphs = [
                    word_chunk
                ]

            continue


        # ====================================================
        # Try adding paragraph
        # ====================================================

        if current_chunk:

            candidate = (
                current_chunk
                + "\n\n"
                + paragraph
            )

        else:

            candidate = paragraph


        # ====================================================
        # Paragraph fits
        # ====================================================

        if len(candidate) <= max_chars:

            current_chunk = candidate

            current_paragraphs.append(
                paragraph
            )


        # ====================================================
        # Paragraph doesn't fit
        # ====================================================

        else:

            # Save current chunk

            if current_chunk:

                chunks.append(
                    current_chunk
                )


            # ------------------------------------------------
            # Create overlap
            # ------------------------------------------------

            overlap_list = (
                current_paragraphs[
                    -overlap_paragraphs:
                ]
                if overlap_paragraphs > 0
                else []
            )

            overlap = "\n\n".join(
                overlap_list
            )


            # ------------------------------------------------
            # Start new chunk
            # ------------------------------------------------

            if overlap:

                current_chunk = (
                    overlap
                    + "\n\n"
                    + paragraph
                ).strip()

                current_paragraphs = (
                    overlap_list.copy()
                )

                current_paragraphs.append(
                    paragraph
                )

            else:

                current_chunk = paragraph

                current_paragraphs = [
                    paragraph
                ]


    # ========================================================
    # Save final chunk
    # ========================================================

    if current_chunk:

        chunks.append(
            current_chunk
        )

    return chunks


# ============================================================
# Create chunk records
# ============================================================

def create_chunks(
    sections,
    document_metadata,
    source_path,
):

    records = []

    global_chunk_number = 0


    for section in sections:

        section_name = (
            section["section"]
        )

        subsection_name = (
            section["subsection"]
        )


        policy_metadata = (
            detect_policy_metadata(
                section_name or "",
                subsection_name or "",
                document_metadata,
            )
        )


        chunks = split_into_chunks(
            section["text"]
        )


        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            global_chunk_number += 1


            chunk_id = (
                f"{document_metadata['doc_id']}"
                f"-CHUNK-"
                f"{global_chunk_number:04d}"
            )


            metadata = {

                # ------------------------------------------------
                # Document
                # ------------------------------------------------

                "doc_id": str(
                    document_metadata.get(
                        "doc_id",
                        "",
                    )
                ),

                "doc_title": str(
                    document_metadata.get(
                        "doc_title",
                        "",
                    )
                ),

                "document_version": str(
                    document_metadata.get(
                        "doc_version",
                        "",
                    )
                ),

                "source": str(
                    source_path
                ),


                # ------------------------------------------------
                # Policy
                # ------------------------------------------------

                "policy_version": str(
                    policy_metadata[
                        "policy_version"
                    ]
                ),

                "policy_status": str(
                    policy_metadata[
                        "policy_status"
                    ]
                ),

                "is_active": bool(
                    policy_metadata[
                        "is_active"
                    ]
                ),

                "effective_date": str(
                    policy_metadata[
                        "effective_date"
                    ] or ""
                ),

                "effective_until": str(
                    policy_metadata[
                        "effective_until"
                    ] or ""
                ),


                # ------------------------------------------------
                # Document structure
                # ------------------------------------------------

                "section": str(
                    section_name or ""
                ),

                "subsection": str(
                    subsection_name or ""
                ),


                # ------------------------------------------------
                # Chunk
                # ------------------------------------------------

                "chunk_id": chunk_id,

                "chunk_number": chunk_number,
            }


            records.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata,
                }
            )


    return records


# ============================================================
# Embed chunks (via API instead of local model)
# ============================================================

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call the embeddings API in batches and return a flat list of vectors."""

    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]

        print(
            f"  Embedding batch {i // EMBED_BATCH_SIZE + 1} "
            f"({len(batch)} texts)..."
        )

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        all_embeddings.extend(
            item.embedding for item in response.data
        )

    return all_embeddings


def embed_chunks(
    records,
):

    texts = [
        record["text"]
        for record in records
    ]


    if not texts:

        return []


    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks via API..."
    )


    embeddings = embed_texts(texts)


    return embeddings


# ============================================================
# Store chunks in ChromaDB
# ============================================================

def store_chunks(
    records,
    embeddings,
):

    if not records:
        return


    ids = [
        record["id"]
        for record in records
    ]


    documents = [
        record["text"]
        for record in records
    ]


    metadatas = [
        record["metadata"]
        for record in records
    ]


    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,  # already plain lists from the API
        metadatas=metadatas,
    )


    print(
        f"Stored {len(records)} chunks "
        f"in ChromaDB."
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Reading document: "
        f"{DOCUMENT_PATH}"
    )


    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    text = read_document(
        DOCUMENT_PATH
    )


    # --------------------------------------------------------
    # Parse front matter
    # --------------------------------------------------------

    document_metadata, lines = (
        parse_front_matter(text)
    )


    print(
        "\nDocument metadata:"
    )

    print(
        document_metadata
    )


    # --------------------------------------------------------
    # Parse sections
    # --------------------------------------------------------

    sections = parse_sections(
        lines
    )


    print(
        f"\nFound {len(sections)} "
        f"content sections."
    )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    records = create_chunks(
        sections,
        document_metadata,
        DOCUMENT_PATH,
    )


    print(
        f"Created {len(records)} chunks."
    )


    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print(
        "\nExample chunks:\n"
    )


    for record in records[:3]:

        print(
            "----------------------------------------"
        )

        print(
            f"ID: {record['id']}"
        )

        print(
            f"Section: "
            f"{record['metadata']['section']}"
        )

        print(
            f"Subsection: "
            f"{record['metadata']['subsection']}"
        )

        print(
            f"Policy version: "
            f"{record['metadata']['policy_version']}"
        )

        print(
            f"Policy status: "
            f"{record['metadata']['policy_status']}"
        )

        print(
            f"Active: "
            f"{record['metadata']['is_active']}"
        )

        print(
            f"Text:\n"
            f"{record['text'][:500]}"
        )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embed_chunks(
        records
    )


    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    store_chunks(
        records,
        embeddings
    )


    print(
        "\nIngestion completed successfully."
    )

    return collection

if __name__ == "__main__":
    main()