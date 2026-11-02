import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data_preprocessing import load_uiprmd_data

class RehabWindowDataset(Dataset):
    def __init__(self, X_list, y_list, window_size=30, step_size=5):
        self.windows = []
        self.labels = []
        for seq, label in zip(X_list, y_list):
            n_frames = len(seq)
            if n_frames < window_size:
                pad = np.pad(seq, ((0, window_size - n_frames), (0, 0)), mode='edge')
                self.windows.append(pad)
                self.labels.append(label)
            else:
                for start in range(0, n_frames - window_size + 1, step_size):
                    self.windows.append(seq[start:start + window_size])
                    self.labels.append(label)
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def get_dataloaders(data_dir=None, window_size=30, step_size=5, batch_size=16, val_split=0.2, seed=42):
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    X_list, y_list, class_names = load_uiprmd_data(data_dir)
    dataset = RehabWindowDataset(X_list, y_list, window_size, step_size)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)

    train_idx = train_ds.indices
    train_windows = dataset.windows[train_idx]
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True) + 1e-7
    dataset.windows = (dataset.windows - mean) / std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    n_features = dataset.windows.shape[-1]
    n_classes = len(class_names)
    return train_loader, val_loader, n_features, n_classes, class_names
