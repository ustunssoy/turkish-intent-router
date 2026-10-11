# Türkçe Niyet Yönlendirici (Intent Router)

BERTürk tabanlı niyet sınıflandırıcı + agentic RAG hattı + Captum açıklanabilirlik.

## Mimari

```
Kullanıcı Sorgusu
       │
       ▼
┌──────────────┐
│   BERTürk    │
│ Sınıflandırıcı│
└──────┬───────┘
       │ niyet tahmini
       ▼
┌──────────────┐
│ Intent Router│
└──┬───┬───┬───┘
   │   │   │
   ▼   ▼   ▼
 ┌───┐┌───┐┌───┐
 │ VS ││ VS ││ VS │  ← FAISS Vektör Depoları (niyet başına)
 └─┬─┘└─┬─┘└─┬─┘
   │    │    │
   ▼    ▼    ▼
  Bağlamsal Dokümanlar
       │
       ▼
     Yanıt
```

## Niyet Sınıfları

| ID | Niyet | Açıklama |
|----|-------|----------|
| 0 | bilgi_sorgulama | Ürün/hizmet bilgisi |
| 1 | siparis_takibi | Kargo/sipariş durumu |
| 2 | sikayet | Şikâyet/memnuniyetsizlik |
| 3 | iptal_iade | İptal/iade/değişim |
| 4 | teknik_destek | Teknik sorun/destek |
| 5 | genel_sohbet | Selamlaşma/sohbet |

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
# 1. Veri seti oluştur
python data/generate_dataset.py

# 2. Modeli eğit
python train.py

# 3. Tahmin
python inference.py

# 4. RAG hattı
python intent_router.py

# 5. Açıklanabilirlik analizi
python explainability.py
```

## Proje Yapısı

```
IntentRouter/
├── config.py              # Konfigürasyon
├── train.py               # BERTürk eğitimi
├── inference.py            # Tahmin modülü
├── intent_router.py        # Agentic RAG hattı
├── explainability.py       # Captum atıf analizi
├── knowledge_base.py       # Bilgi bankası
├── data/
│   └── generate_dataset.py # Veri seti oluşturucu
├── models/                 # Eğitilmiş modeller
├── results/                # Sonuçlar ve grafikler
└── requirements.txt
```

## Teknolojiler

- **PyTorch** + **Transformers**: Model eğitimi
- **BERTürk** (`dbmdz/bert-base-turkish-cased`): Türkçe dil modeli
- **LangChain** + **FAISS**: Agentic RAG hattı
- **Captum**: Token düzeyinde açıklanabilirlik
- **Sentence-Transformers**: Embedding üretimi
