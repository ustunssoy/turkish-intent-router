"""LangChain + FAISS tabanlı agentic RAG hattı – niyet yönlendirmeli."""

import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

sys.path.append(str(Path(__file__).parent))
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, INTENT_LABELS, TOP_K
from inference import IntentClassifier
from knowledge_base import KNOWLEDGE_BASE


class IntentRouter:
    """Niyet tabanlı sorgu yönlendirici (agentic RAG)."""

    def __init__(self, classifier: IntentClassifier | None = None):
        self.classifier = classifier or IntentClassifier()

        # Embedding modeli
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )

        # Her niyet için ayrı vektör deposu oluştur
        self.vector_stores: dict[str, FAISS] = {}
        self._build_vector_stores()

    def _build_vector_stores(self):
        """Her niyet sınıfı için FAISS vektör deposu oluştur."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )

        for intent_name, docs in KNOWLEDGE_BASE.items():
            documents = []
            for doc in docs:
                chunks = splitter.split_text(doc["content"])
                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "intent": intent_name,
                                "title": doc["title"],
                            },
                        )
                    )

            if documents:
                self.vector_stores[intent_name] = FAISS.from_documents(
                    documents, self.embeddings
                )

        print(f"{len(self.vector_stores)} niyet kolu için vektör depoları oluşturuldu.")

    def route(self, query: str) -> dict:
        """Sorguyu sınıflandır ve ilgili bilgi koluna yönlendir."""
        # 1. Niyet sınıflandırma
        intent_result = self.classifier.predict(query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # 2. İlgili vektör deposundan arama
        context_docs = []
        if intent in self.vector_stores:
            results = self.vector_stores[intent].similarity_search_with_score(
                query, k=TOP_K
            )
            context_docs = [
                {
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "score": float(score),
                }
                for doc, score in results
            ]

        # 3. Yanıt oluştur
        context_text = "\n\n".join(
            [f"[{d['title']}]: {d['content']}" for d in context_docs]
        )

        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "all_probs": intent_result["all_probs"],
            "retrieved_docs": context_docs,
            "context": context_text,
            "routing_info": {
                "vector_store": intent,
                "num_docs_retrieved": len(context_docs),
            },
        }

    def route_batch(self, queries: list[str]) -> list[dict]:
        """Toplu yönlendirme."""
        return [self.route(q) for q in queries]


def format_response(result: dict) -> str:
    """Yönlendirme sonucunu okunabilir formata çevir."""
    lines = [
        f"{'='*60}",
        f"Sorgu    : {result['query']}",
        f"Niyet    : {result['intent']} (güven: {result['confidence']:.3f})",
        f"Yönlendirme: {result['routing_info']['vector_store']} koluna yönlendirildi",
        f"Bulunan doküman: {result['routing_info']['num_docs_retrieved']}",
        f"{'-'*60}",
        "Bağlam:",
    ]

    for doc in result["retrieved_docs"]:
        lines.append(f"  [{doc['title']}] (skor: {doc['score']:.4f})")
        lines.append(f"  {doc['content'][:150]}...")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    router = IntentRouter()

    test_queries = [
        "Bu ürünün fiyatı ne kadar?",
        "Kargom nerede?",
        "Ürün kırık geldi, çok memnuniyetsizim!",
        "Siparişimi iptal etmek istiyorum.",
        "Telefon açılmıyor ne yapmalıyım?",
        "Merhaba, nasılsınız?",
    ]

    for query in test_queries:
        result = router.route(query)
        print(format_response(result))
        print()
