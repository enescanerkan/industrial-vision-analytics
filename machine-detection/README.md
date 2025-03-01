# Machine Detection System

Bu proje, video üzerinde makine tespiti yapan ve belirlenen bölgelerdeki makine sayımını gerçekleştiren bir sistemdir.

## 🚀 Özellikler

- YOLO tabanlı makine tespiti
- ByteTrack ile nesne takibi
- Özelleştirilebilir bölge (zone) tanımlama
- Bölge bazlı sayım
- Gerçek zamanlı görselleştirme

## 📋 Gereksinimler

Projeyi çalıştırmak için aşağıdaki gereksinimlere ihtiyacınız vardır:

```bash
numpy>=1.21.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
supervision>=0.18.0
```

## 🛠️ Kurulum

1. Projeyi klonlayın:
```bash
git clone <repo-url>
cd machine-detection
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. YOLO modelini `models` klasörüne yerleştirin.

## 📝 Konfigürasyon

`src/config.py` dosyasında aşağıdaki ayarları yapılandırabilirsiniz:

- `BASE_DIR`: Proje ana dizini
- `MODEL_PATH`: YOLO model dosyasının yolu
- `VIDEO_PATH`: İşlenecek video dosyasının yolu
- `ZONES_PATH`: Bölge tanımlamalarının bulunduğu JSON dosyasının yolu
- `CONFIDENCE_THRESHOLD`: Tespit güven eşiği
- `IOU_THRESHOLD`: IoU eşik değeri

## 🎯 Kullanım

### 1. Bölge Tanımlama

Bölgeleri tanımlamak için:

```bash
python zones/zone_selector.py
```

Kullanım:
- Sol tık ile çizmeye başlayın
- Sürükleyerek bölgeyi belirleyin
- Bırakın
- 'q' tuşu ile çıkın
- 's' tuşu ile kaydedin
- 'r' tuşu ile son bölgeyi silin

### 2. Makine Tespiti ve Sayımı

Ana programı çalıştırmak için:

```bash
python src/main.py
```

## 📁 Proje Yapısı

```
machine-detection/
├── src/
│   ├── config.py        # Konfigürasyon ayarları
│   ├── main.py          # Ana program
│   ├── detect.py        # Tespit işlemleri
│   └── zone_counter.py  # Bölge sayım mantığı
├── zones/
│   ├── zone_selector.py # Bölge seçim aracı
│   └── zones.json       # Bölge tanımlamaları
├── models/
│   ├── best_wc.pt       # YOLO model dosyası
│   └── bytetrack.yaml   # ByteTrack konfigürasyonu
└── requirements.txt
```

## 🔍 Önemli Notlar

- Sistem CUDA destekli GPU varsa otomatik olarak GPU'yu kullanacaktır
- Bölge tanımlamaları farklı video boyutları için otomatik olarak ölçeklendirilir
- ByteTrack ile nesne takibi yapılarak daha tutarlı sayım sağlanır

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje [MIT](LICENSE) lisansı altında lisanslanmıştır.
