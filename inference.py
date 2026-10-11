"""Eğitilmiş BERTürk modeli ile niyet tahmini."""

import sys
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.append(str(Path(__file__).parent))
from config import INTENT_LABELS, MAX_LENGTH, MODEL_DIR


class IntentClassifier:
    """Niyet sınıflandırıcı."""

    def __init__(self, model_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_path or (MODEL_DIR / "best_model")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Tek metin için niyet tahmini."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        return {
            "intent": INTENT_LABELS[pred_id],
            "intent_id": pred_id,
            "confidence": confidence,
            "all_probs": {
                INTENT_LABELS[i]: probs[0][i].item() for i in range(len(INTENT_LABELS))
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Toplu tahmin."""
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    classifier = IntentClassifier()

    test_queries = [
        "Bu ürünün fiyatı nedir?",
        "Kargom nerede kaldı?",
        "Ürün bozuk geldi, çok kızgınım!",
        "Siparişimi iptal etmek istiyorum.",
        "Cihaz açılmıyor ne yapmalıyım?",
        "Merhaba, nasılsınız?",
    ]

    for q in test_queries:
        result = classifier.predict(q)
        print(f"Sorgu: {q}")
        print(f"  Niyet: {result['intent']} (güven: {result['confidence']:.3f})")
        print()
