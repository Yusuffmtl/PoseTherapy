# Fizik Tedavi ve Rehabilitasyon Hareketleri Siniflandirma

Bu projede UI-PRMD (University of Idaho Physical Rehabilitation Movement Dataset) veri seti kullanilarak fizik tedavi ve rehabilitasyon egzersizlerinin LSTM ve 1D-CNN derin ogrenme modelleri ile siniflandirilmasi amaclanmistir. Ayrica OpenCV ve MediaPipe kullanilarak web kamerasindan canli hareket takibi gerceklestirilmistir.

## Hareket Siniflari (UI-PRMD)
1. Derin Squat (Rehabilitasyon)
2. Engel Adimlama (Hurdle Step)
3. Duz Cizgi Lunge (Inline Lunge)
4. Yan Lunge (Side Lunge)
5. Otur-Kalk (Sit to Stand)
6. Bacak Kaldirma (Straight Leg Raise)
7. Omuz Abduksiyon (Kolu Yana Kaldirma)
8. Omuz Ekstansiyon
9. Omuz Rotasyonu
10. Omuz Skapsiyon

## Dosya Yapisi
- `src/data_preprocessing.py`: UI-PRMD txt dosyalarini okuma ve gurultu filtreleme fonksiyonlari.
- `src/dataset.py`: Pencereli (sliding window) PyTorch Dataset ve DataLoader yapisi.
- `src/models/lstm_model.py`: Cift yonlu LSTM modeli.
- `src/models/cnn_model.py`: 1-Boyutlu Evrisimli Sinir Agi (1D-CNN) modeli.
- `train.py`: Model egitim scripti.
- `evaluate.py`: Dogruluk ve F1 skoru hesaplama scripti.
- `live_camera.py`: Web kamerasindan canli iskelet takibi, tekrar sayma ve hareket geri bildirimi.
- `predict.py`: Tekil bir hareket dosyasi uzerinden tahmin yapma.
- `visualize.py`: Karmasiklik matrisi grafigi olusturma.

## Kurulum
```bash
pip install -r requirements.txt
```

## Calistirma

Modeli egitmek icin:
```bash
python train.py --model lstm
python train.py --model cnn
```

Egitilen modeli test etmek icin:
```bash
python evaluate.py --model lstm
```

Kameradan canli hareket takibi yapmak icin:
```bash
python live_camera.py
```
