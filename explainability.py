"""Captum ile token atıf (attribution) analizi."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients, visualization
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.append(str(Path(__file__).parent))
from config import INTENT_LABELS, MAX_LENGTH, MODEL_DIR, RESULTS_DIR


class IntentExplainer:
    """Captum tabanlı model açıklayıcı."""

    def __init__(self, model_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_path or (MODEL_DIR / "best_model")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Layer Integrated Gradients – embedding katmanı üzerinde
        self.lig = LayerIntegratedGradients(
            self._forward_func, self.model.bert.embeddings
        )

    def _forward_func(self, input_ids, attention_mask):
        """Model ileri geçiş."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def explain(self, text: str, target: int | None = None, n_steps: int = 50) -> dict:
        """Token düzeyinde atıf analizi."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Hedef sınıf (belirtilmezse tahmin edilen sınıf)
        if target is None:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                target = torch.argmax(outputs.logits, dim=-1).item()

        # Atıf hesapla
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Token başına atıf skoru (embedding boyutu üzerinde toplama)
        attr_scores = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attr_scores / torch.norm(attr_scores)
        attr_scores = attr_scores.cpu().detach().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # Özel token'ları filtrele
        token_attrs = []
        for token, score in zip(tokens, attr_scores):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                token_attrs.append({"token": token, "attribution": float(score)})

        return {
            "text": text,
            "predicted_intent": INTENT_LABELS[target],
            "target_class": target,
            "token_attributions": token_attrs,
            "convergence_delta": float(delta.item()),
        }

    def visualize(self, text: str, target: int | None = None, save_path: str | None = None):
        """Atıf görselleştirmesi."""
        result = self.explain(text, target)

        tokens = [ta["token"] for ta in result["token_attributions"]]
        scores = [ta["attribution"] for ta in result["token_attributions"]]

        fig, ax = plt.subplots(figsize=(max(len(tokens) * 0.8, 8), 4))

        colors = ["#e74c3c" if s < 0 else "#2ecc71" for s in scores]
        bars = ax.bar(range(len(tokens)), scores, color=colors, edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Atıf Skoru", fontsize=12)
        ax.set_title(
            f"Token Atıf Analizi\n"
            f"Metin: \"{text}\"\n"
            f"Tahmin: {result['predicted_intent']}",
            fontsize=12,
            pad=15,
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Kaydedildi: {save_path}")
        else:
            save_path = RESULTS_DIR / "attribution_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Kaydedildi: {save_path}")

        plt.close()
        return result

    def batch_explain(self, texts: list[str]):
        """Birden fazla metin için atıf analizi ve karşılaştırma."""
        results = []
        for text in texts:
            result = self.explain(text)
            results.append(result)

        # Özet görselleştirme
        fig, axes = plt.subplots(
            len(texts), 1, figsize=(12, 3 * len(texts)), squeeze=False
        )

        for i, (result, ax) in enumerate(zip(results, axes[:, 0])):
            tokens = [ta["token"] for ta in result["token_attributions"]]
            scores = [ta["attribution"] for ta in result["token_attributions"]]

            colors = ["#e74c3c" if s < 0 else "#2ecc71" for s in scores]
            ax.bar(range(len(tokens)), scores, color=colors)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_title(
                f"{result['predicted_intent']} | \"{result['text'][:50]}\"",
                fontsize=10,
            )
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        save_path = RESULTS_DIR / "batch_attribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Toplu analiz kaydedildi: {save_path}")

        return results


if __name__ == "__main__":
    explainer = IntentExplainer()

    test_texts = [
        "Bu ürünün fiyatı nedir?",
        "Siparişim nerede kaldı?",
        "Ürün bozuk geldi çok kızgınım!",
        "İade etmek istiyorum.",
        "Cihaz açılmıyor.",
        "Merhaba nasılsınız?",
    ]

    for text in test_texts:
        result = explainer.visualize(text)
        print(f"\nMetin: {text}")
        print(f"Tahmin: {result['predicted_intent']}")
        print("En etkili tokenlar:")
        sorted_attrs = sorted(
            result["token_attributions"],
            key=lambda x: abs(x["attribution"]),
            reverse=True,
        )
        for ta in sorted_attrs[:5]:
            print(f"  {ta['token']:15s} → {ta['attribution']:+.4f}")

    explainer.batch_explain(test_texts)
