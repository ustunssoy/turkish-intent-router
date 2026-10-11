"""Sentetik Türkçe niyet veri seti oluşturma."""

import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, SEED

random.seed(SEED)

TEMPLATES = {
    "bilgi_sorgulama": [
        "Bu ürünün özellikleri nelerdir?",
        "Kargo ücreti ne kadar?",
        "Mağaza çalışma saatleri nedir?",
        "Taksit seçenekleri var mı?",
        "Garanti süresi ne kadar?",
        "Bu ürün stokta var mı?",
        "Fiyat bilgisi alabilir miyim?",
        "Ödeme yöntemleri nelerdir?",
        "Üyelik avantajları nedir?",
        "Kampanya detaylarını öğrenebilir miyim?",
        "Hangi renk seçenekleri mevcut?",
        "Teslimat süresi ne kadar?",
        "Minimum sipariş tutarı var mı?",
        "Ürün boyutları hakkında bilgi verir misiniz?",
        "İndirim kuponu nasıl kullanılır?",
        "Ücretsiz kargo limiti nedir?",
        "Bu ürünün malzeme bilgisi nedir?",
        "Hediye paketi seçeneği var mı?",
        "Sadakat programınız hakkında bilgi alabilir miyim?",
        "Mağaza iade politikası nedir?",
        "Kapıda ödeme seçeneği var mı?",
        "Bu ürün hangi markaya ait?",
        "Ürün menşei neresidir?",
        "Beden tablosu mevcut mu?",
        "Toptan satış yapıyor musunuz?",
        "Fatura bilgisi nasıl düzenlenir?",
        "Bayilik şartları nelerdir?",
        "Kurumsal satış departmanına nasıl ulaşabilirim?",
        "Ürün sertifikaları hakkında bilgi verir misiniz?",
        "Aynı gün teslimat var mı?",
    ],
    "siparis_takibi": [
        "Siparişim nerede?",
        "Kargom ne zaman gelecek?",
        "Sipariş durumumu öğrenmek istiyorum.",
        "Takip numaram nedir?",
        "Paketim teslim edildi mi?",
        "Kargom neden gecikiyor?",
        "Siparişim onaylandı mı?",
        "Teslimat tarihi ne zaman?",
        "Kargo firması hangisi?",
        "Siparişimin kargo takip numarasını verir misiniz?",
        "Paketim şu an nerede?",
        "Sipariş geçmişimi görmek istiyorum.",
        "Son siparişim hakkında bilgi alabilir miyim?",
        "Kargom dağıtıma çıktı mı?",
        "Teslimat adresimi değiştirebilir miyim?",
        "Siparişim hazırlanıyor mu?",
        "Kargo ücreti iadesi yapılacak mı?",
        "Eksik ürün geldi, siparişimi kontrol eder misiniz?",
        "Yanlış ürün gönderilmiş, sipariş numaramı paylaşıyorum.",
        "Siparişim iptal mi edildi?",
        "İki gün önce verdiğim sipariş hâlâ kargoya verilmedi.",
        "Kargo şubeye mi bırakıldı?",
        "Teslim alan kişi bilgisi nedir?",
        "Kurye ile iletişime geçebilir miyim?",
        "Farklı bir kargo şubesinden teslim alabilir miyim?",
        "Siparişim gümrükte mi bekliyor?",
        "Yurt dışı kargo takibi nasıl yapılır?",
        "Kargom hasarlı geldi, teslimat bilgisini kontrol eder misiniz?",
        "Toplam kaç adet ürün gönderildi?",
        "Kargonun tahmini varış süresi nedir?",
    ],
    "sikayet": [
        "Ürün bozuk geldi!",
        "Müşteri hizmetlerine ulaşamıyorum.",
        "Çok kötü bir deneyim yaşadım.",
        "Ürün kalitesinden memnun değilim.",
        "Siparişim çok geç geldi.",
        "Ambalaj hasarlıydı.",
        "Ürün fotoğraftakiyle aynı değil.",
        "Fiyat haksız yere yükseltilmiş.",
        "Müşteri temsilcisi ilgisizdi.",
        "Defalarca aradım ama sorunum çözülmedi.",
        "Ürün açıldığında kırık çıktı.",
        "Hizmet kalitesinden şikayetçiyim.",
        "Reklamı yapılan ürünle gelen ürün farklı.",
        "Web sitesinde yanlış bilgi var.",
        "Ödeme sırasında hata oluştu ve param çekildi.",
        "Garanti kapsamında olması gereken ürün reddedildi.",
        "Mağaza personeli çok kabaydı.",
        "Aynı sorunu üçüncü kez yaşıyorum.",
        "Vaat edilen kampanya uygulanmadı.",
        "Yetkili servis randevusu alamıyorum.",
        "Ürünün son kullanma tarihi geçmişti.",
        "Uygulama üzerinden işlem yapamıyorum, sürekli hata veriyor.",
        "Taahhüt edilen hediye gelmedi.",
        "Fatura tutarı sipariş tutarından farklı.",
        "Kredi kartıma çift çekim yapılmış.",
        "Gizlilik politikanızı ihlal ettiniz.",
        "Sosyal medyadan yazdım ama cevap alamadım.",
        "Teslimat sırasında ürüne zarar verilmiş.",
        "Canlı destek hattı hiç çalışmıyor.",
        "İade sürecim çok uzun sürüyor.",
    ],
    "iptal_iade": [
        "Siparişimi iptal etmek istiyorum.",
        "İade sürecini başlatmak istiyorum.",
        "Ürünü geri göndermek istiyorum.",
        "Para iadem ne zaman yapılır?",
        "İade koşulları nelerdir?",
        "Değişim yapmak istiyorum.",
        "Siparişimi henüz iptal edebilir miyim?",
        "İade kargo ücreti bana mı ait?",
        "Ürün değişimi nasıl yapılır?",
        "İade formunu nereden bulabilirim?",
        "Cayma hakkımı kullanmak istiyorum.",
        "İade onayım ne zaman gelir?",
        "Ürünü kullanmadım, iade edebilir miyim?",
        "Hediye olarak aldığım ürünü iade edebilir miyim?",
        "Para iadesi hangi hesaba yapılacak?",
        "Kısmi iade yapılabilir mi?",
        "İade sürem doldu mu?",
        "Farklı beden ile değişim istiyorum.",
        "Hatalı ürün iadesi için ne yapmalıyım?",
        "İade kargosunu hangi firma ile göndermeliyim?",
        "İade talebim reddedildi, neden?",
        "Fatura olmadan iade yapılır mı?",
        "İade durumumu nasıl takip ederim?",
        "Açılmış ürünü iade edebilir miyim?",
        "Taksitli alışverişte iade nasıl işler?",
        "Kampanyalı ürünü iade edersem indirim iptal olur mu?",
        "Set halinde aldığım üründen birini iade edebilir miyim?",
        "İade işlemim onaylandı mı?",
        "Para iadesi kredi kartıma ne zaman yansır?",
        "Değişim için ek ücret ödenir mi?",
    ],
    "teknik_destek": [
        "Ürün çalışmıyor, ne yapmalıyım?",
        "Kurulum nasıl yapılır?",
        "Şifre sıfırlama işlemi yapamıyorum.",
        "Uygulama sürekli çöküyor.",
        "Yazılım güncelleme hatası alıyorum.",
        "Bluetooth bağlantısı kurulamıyor.",
        "Ekran donuyor, çözüm nedir?",
        "Pil çok çabuk bitiyor.",
        "Wi-Fi bağlantı sorunu yaşıyorum.",
        "Cihaz açılmıyor.",
        "Ses gelmiyor, hoparlör arızalı mı?",
        "Kamera çalışmıyor.",
        "Yazıcı bağlantı hatası veriyor.",
        "Dosya aktarımı yapamıyorum.",
        "Hesabıma giriş yapamıyorum.",
        "İki faktörlü doğrulama kodu gelmiyor.",
        "Uygulamada bildirim almıyorum.",
        "Cihaz aşırı ısınıyor.",
        "Ekran görüntüsü nasıl alınır?",
        "Fabrika ayarlarına döndürme nasıl yapılır?",
        "USB port çalışmıyor.",
        "Ses kaydı nasıl yapılır?",
        "VPN bağlantısı kuramıyorum.",
        "Depolama alanı yetersiz uyarısı alıyorum.",
        "Otomatik güncellemeyi nasıl kapatırım?",
        "Yazılım sürümünü nasıl kontrol ederim?",
        "Sistem gereksinimleri nelerdir?",
        "Uyumluluk sorunu yaşıyorum.",
        "Veri yedekleme nasıl yapılır?",
        "Driver yükleme hatası alıyorum.",
    ],
    "genel_sohbet": [
        "Merhaba, nasılsınız?",
        "İyi günler!",
        "Teşekkür ederim.",
        "Görüşmek üzere.",
        "Bugün hava çok güzel.",
        "Ne yapıyorsunuz?",
        "İyi akşamlar.",
        "Kolay gelsin.",
        "Sağ olun, iyi çalışmalar.",
        "Hayırlı işler.",
        "Günaydın!",
        "İyi geceler.",
        "Bir şey sormak istiyorum ama konuyu unuttum.",
        "Şaka yapıyorum, merak etmeyin.",
        "Canınız sıkılmıyor mu?",
        "Robot musunuz yoksa gerçek bir insan mı?",
        "Yapay zeka hakkında ne düşünüyorsunuz?",
        "Türkçeyi çok iyi konuşuyorsunuz.",
        "Adınız ne?",
        "Kendinizi tanıtır mısınız?",
        "Kaç yaşındasınız?",
        "Favori renginiz ne?",
        "Bugün beni güldürdünüz, teşekkürler.",
        "Çok yardımcı oldunuz.",
        "Size bir iltifat etmek isterim.",
        "Başka bir konuda da yardımcı olur musunuz?",
        "Biraz sohbet edelim mi?",
        "Hava durumu nasıl olacak?",
        "Güzel bir gün geçirmenizi dilerim.",
        "Hoşça kalın.",
    ],
}


def augment(text: str) -> list[str]:
    """Basit veri çoğaltma."""
    variants = [text]

    # Küçük/büyük harf varyasyonu
    variants.append(text.lower())
    variants.append(text.upper())

    # Noktalama kaldırma
    cleaned = text.replace("?", "").replace("!", "").replace(".", "").replace(",", "")
    if cleaned != text:
        variants.append(cleaned)

    # Kelime tekrarı ekleme
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        new_words = words[:idx] + [words[idx]] + words[idx:]
        variants.append(" ".join(new_words))

    return variants


def generate_dataset():
    """Veri setini oluştur ve kaydet."""
    samples = []

    for label, texts in TEMPLATES.items():
        for text in texts:
            augmented = augment(text)
            for aug_text in augmented:
                samples.append({"text": aug_text, "label": label})

    random.shuffle(samples)

    output_path = DATA_DIR / "intent_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Toplam örnek: {len(samples)}")
    print(f"Kaydedildi: {output_path}")

    # İstatistik
    from collections import Counter
    label_counts = Counter(s["label"] for s in samples)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    return samples


if __name__ == "__main__":
    generate_dataset()
