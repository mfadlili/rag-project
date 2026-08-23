import chromadb

from config import CHROMA_DIR

from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "nusantaracare_documents"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def main():
    print(f"Opening ChromaDB: {CHROMA_DIR}")

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collections = chroma_client.list_collections()

    print("\nCollections:")
    for collection in collections:
        print(f"- {collection.name}")

    try:
        collection = chroma_client.get_collection(
            COLLECTION_NAME
        )
    except Exception:
        print(
            f"\nCollection '{COLLECTION_NAME}' "
            "does not exist."
        )
        return

    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Document count: {collection.count()}")

    if collection.count() == 0:
        print("Collection is empty.")
        return

    # Get all documents
    results = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )


    question = "email pada kondisi darurat"

    # ubah pertanyaan menjadi embed
    model = SentenceTransformer(
    EMBEDDING_MODEL
)
    q_embed = model.encode(question).tolist()
    answer = collection.query(
        query_embeddings=[q_embed],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )


    print('ini answer')

    print(answer)


    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    print(f"\nRetrieved {len(ids)} records.\n")

    for i, (
        record_id,
        document,
        metadata,
    ) in enumerate(
        zip(ids, documents, metadatas),
        start=1,
    ):

        print("=" * 80)
        print(f"Record #{i}")
        print(f"ID: {record_id}")

        print("\nMetadata:")
        print(metadata)

        print("\nDocument:")
        print(document)

    print("=" * 80)


if __name__ == "__main__":
    main()