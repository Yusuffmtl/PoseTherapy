import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_preprocessing import load_uiprmd_data

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    X_list, y_list, class_names = load_uiprmd_data(data_dir)

    counts = [0] * len(class_names)
    for y in y_list:
        counts[y] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, counts, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Ornek Sayisi')
    plt.title('UI-PRMD Veri Seti Sinif Dagilimi')
    plt.tight_layout()
    plt.savefig('eda_class_distribution.png')
    plt.close()
    print("Kaydedildi: eda_class_distribution.png")

    nan_counts = []
    files = sorted(glob.glob(os.path.join(data_dir, '*_angles.txt')))
    for f in files:
        df = pd.read_csv(f, header=None)
        nan_counts.append(df.isna().sum().sum())

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(nan_counts) + 1), nan_counts, color='coral')
    plt.xlabel('Dosya Indeksi')
    plt.ylabel('Eksik Deger Sayisi')
    plt.title('Dosyalardaki Eksik Deger (NaN) Dagilimi')
    plt.tight_layout()
    plt.savefig('eda_missing_values.png')
    plt.close()
    print("Kaydedildi: eda_missing_values.png")

if __name__ == '__main__':
    main()
