import os
import glob
import numpy as np
import pandas as pd

def load_uiprmd_file(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split(',')])
    return np.array(rows, dtype=np.float32)

def parse_filename(filename):
    base = os.path.basename(filename).replace('.txt', '')
    parts = base.split('_')
    m_id = int(parts[0][1:])
    s_id = int(parts[1][1:])
    e_id = int(parts[2][1:])
    d_type = parts[3]
    return m_id, s_id, e_id, d_type

def interpolate_missing_values(arr):
    df = pd.DataFrame(arr)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(0.0)
    return df.values.astype(np.float32)

def apply_moving_average_filter(arr, window_size=5):
    df = pd.DataFrame(arr)
    smoothed = df.rolling(window=window_size, min_periods=1, center=True).mean()
    return smoothed.values.astype(np.float32)

def load_uiprmd_data(data_dir, data_type='angles', apply_filter=True):
    pattern = os.path.join(data_dir, f'*_{data_type}.txt')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{data_type} dosyalari bulunamadi: {data_dir}")

    X_list, y_list = [], []
    for fpath in files:
        m_id, s_id, e_id, _ = parse_filename(fpath)
        data = load_uiprmd_file(fpath)
        data = interpolate_missing_values(data)
        if apply_filter:
            data = apply_moving_average_filter(data)
        X_list.append(data)
        y_list.append(m_id - 1)

    class_names = [
        "Derin Squat",
        "Engel Adimlama",
        "Duz Cizgi Lunge",
        "Yan Lunge",
        "Otur-Kalk",
        "Bacak Kaldirma",
        "Omuz Abduksiyon",
        "Omuz Ekstansiyon",
        "Omuz Rotasyonu",
        "Omuz Skapsiyon"
    ]
    return X_list, y_list, class_names
