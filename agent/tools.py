# tools.py — phiên bản có logging + filter theo metadata động vật

import json
import logging
from langchain_core.tools import tool
from engine.embedding import LocalEmbedder
from engine.retrieval import EmbeddingStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RAG-Tool")

_embedder: LocalEmbedder | None = None
_store: EmbeddingStore | None = None

def _get_embedder() -> LocalEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder()
    return _embedder

def _get_store() -> EmbeddingStore:
    global _store
    if _store is None:
        # FIX 1: Dùng persist_path để data không mất khi restart
        _store = EmbeddingStore(persist_path="./chroma_db")
    return _store


def create_retrieval_tool(collection_name: str = "lab14"):

    @tool
    def retrieve_documents(query: str, animal_filter: str = "") -> str:
        """
        Tìm kiếm tài liệu trong knowledge base.

        - query: câu truy vấn tự nhiên
        - animal_filter: (tuỳ chọn) tên động vật để lọc metadata,
          ví dụ "chó", "mèo", "hổ". Để trống nếu không cần lọc.
        """
        logger.info("=" * 50)
        logger.info(f"[TOOL CALLED] query='{query}' | animal_filter='{animal_filter}'")

        embedder = _get_embedder()
        store = _get_store()

        # Bước 1: Encode query
        logger.info(f"[STEP 1] Encoding query thành vector...")
        query_embedding = embedder.encode(query, prompt_name="web_search_query")
        logger.info(f"[STEP 1] Done — vector dim={len(query_embedding)}")

        # Bước 2: Xây metadata filter nếu có animal_filter
        metadata_filter = None
        if animal_filter.strip():
            metadata_filter = {"animal": animal_filter.strip()}
            logger.info(f"[STEP 2] Áp dụng metadata filter: {metadata_filter}")
        else:
            logger.info(f"[STEP 2] Không có metadata filter — search toàn bộ collection")

        # Bước 3: Search
        logger.info(f"[STEP 3] Searching collection='{collection_name}' top_k=3...")
        if metadata_filter:
            results = store.search_with_filter(
                collection_name, query_embedding, top_k=3,
                metadata_filter=metadata_filter
            )
        else:
            results = store.search(collection_name, query_embedding, top_k=3)

        logger.info(f"[STEP 3] Tìm được {len(results)} kết quả")

        # Bước 4: Format kết quả + log từng doc
        if not results:
            logger.warning("[STEP 4] Không tìm thấy tài liệu nào!")
            return json.dumps(
                {"message": "Không tìm thấy tài liệu liên quan. Hãy thử query hoặc filter khác."},
                ensure_ascii=False,
            )

        docs = []
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            # FIX 2: Lấy doc_id từ đúng field
            doc_id = meta.get("doc_id", meta.get("id", "unknown"))
            doc = {
                "content": r.get("content", ""),
                "score": round(r.get("score", 0.0), 4),
                "doc_id": doc_id,
                "animal": meta.get("animal", ""),
                "title": meta.get("title", ""),
            }
            docs.append(doc)
            logger.info(
                f"[STEP 4] Doc #{i+1}: doc_id={doc_id!r} | "
                f"animal={meta.get('animal', 'N/A')!r} | "
                f"score={doc['score']} | "
                f"preview='{doc['content'][:60]}...'"
            )

        logger.info(f"[DONE] Trả về {len(docs)} docs cho agent")
        logger.info("=" * 50)

        return json.dumps(docs, ensure_ascii=False)

    return retrieve_documents