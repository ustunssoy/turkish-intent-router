"""BERTürk tabanlı niyet sınıflandırıcı – eğitim modülü."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.append(str(Path(__file__).parent))
from config import (
    BATCH_SIZE,
    BERTURK_MODEL_NAME,
    DATA_DIR,
    INTENT_LABELS,
    LABEL_TO_ID,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIR,
    NUM_EPOCHS,
    NUM_LABELS,
    RESULTS_DIR,
    SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntentDataset(Dataset):
    """Niyet veri seti."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_data():
    """Veri setini yükle ve böl."""
    data_path = DATA_DIR / "intent_dataset.json"
    if not data_path.exists():
        from data.generate_dataset import generate_dataset
        generate_dataset()

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    texts = [s["text"] for s in samples]
    labels = [LABEL_TO_ID[s["label"]] for s in samples]

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=(1 - TRAIN_SPLIT), random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train():
    """Modeli eğit."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained(BERTURK_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        BERTURK_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=INTENT_LABELS,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    # Veri
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    train_dataset = IntentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = IntentDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = IntentDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Eğitim döngüsü
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Doğrulama
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch["labels"].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_DIR / "best_model")
            tokenizer.save_pretrained(MODEL_DIR / "best_model")
            print(f"  → En iyi model kaydedildi (acc: {val_acc:.4f})")

    # Test değerlendirmesi
    print("\n" + "=" * 50)
    print("TEST DEĞERLENDİRMESİ")
    print("=" * 50)

    model = BertForSequenceClassification.from_pretrained(MODEL_DIR / "best_model")
    model.to(device)
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch["labels"].cpu().numpy())

    test_acc = accuracy_score(test_true, test_preds)
    print(f"\nTest Doğruluğu: {test_acc:.4f}")
    print("\nSınıflandırma Raporu:")
    target_names = [INTENT_LABELS[i] for i in range(NUM_LABELS)]
    report = classification_report(test_true, test_preds, target_names=target_names)
    print(report)

    # Sonuçları kaydet
    results = {
        "test_accuracy": test_acc,
        "history": history,
        "classification_report": classification_report(
            test_true, test_preds, target_names=target_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(test_true, test_preds).tolist(),
    }

    with open(RESULTS_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return model, tokenizer, results


if __name__ == "__main__":
    train()
