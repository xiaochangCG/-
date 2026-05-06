"""HuBERT 特征分类器 - PyTorch CNN
输入: 768-dim HuBERT 特征 (代替 13-dim MFCC)
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


class HuBERTCNN(nn.Module):
    """1D CNN for HuBERT features (768-dim input)"""
    def __init__(self, in_ch=768):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 768, T) 或 (B, T, 768)
        if x.dim() == 3 and x.shape[-1] == 768:
            x = x.transpose(1, 2)  # (B, T, 768) → (B, 768, T)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = x.mean(dim=-1)  # (B, 64)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x)).squeeze(-1)


class HuBERTClassifier:
    """HuBERT 分类器包装类"""

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = HuBERTCNN()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.threshold = 0.5
        self.X_mean = None
        self.X_std = None
        if self.device.type == 'cuda':
            print(f'Using GPU: {torch.cuda.get_device_name(0)}', flush=True)

    def fit(self, X, y, epochs=100, lr=0.001, batch_size=32, val_split=0.2):
        # X: list of (n_frames, 768) arrays
        # 归一化
        all_frames = np.vstack([f for f in X])
        self.X_mean = all_frames.mean(axis=0, keepdims=True)  # (1, 768)
        self.X_std = all_frames.std(axis=0, keepdims=True) + 1e-8
        X_norm = [(f - self.X_mean) / self.X_std for f in X]

        # 转换为 tensor 并 padding
        max_len = max(f.shape[0] for f in X_norm)

        def to_tensor(batch_X, batch_y=None):
            padded = np.zeros((len(batch_X), max_len, 768), dtype=np.float32)
            for i, f in enumerate(batch_X):
                padded[i, :f.shape[0], :] = f[:max_len]
            t = torch.from_numpy(padded).to(self.device)
            if batch_y is not None:
                return t, torch.from_numpy(batch_y).float().to(self.device)
            return t

        # Split
        if len(X_norm) > 1:
            indices = np.arange(len(X_norm))
            train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42,
                                                   stratify=y if len(set(y)) > 1 else None)
            X_train = [X_norm[i] for i in train_idx]
            y_train = y[train_idx]
            X_val = [X_norm[i] for i in val_idx]
            y_val = y[val_idx]
        else:
            X_train, y_train = X_norm, y
            X_val, y_val = [], np.array([])

        opt = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            self.model.train()
            perm = np.random.permutation(len(X_train))
            total_loss = 0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = [X_train[j] for j in batch_idx]
                y_batch = y_train[batch_idx]
                X_t, y_t = to_tensor(X_batch, y_batch)

                opt.zero_grad()
                preds = self.model(X_t)
                loss = criterion(preds, y_t)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            # Validation
            val_loss = 0
            val_acc = 0
            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    X_t, y_t = to_tensor(X_val, y_val)
                    val_preds = self.model(X_t)
                    val_loss = criterion(val_preds, y_t).item()
                    val_acc = ((val_preds > 0.5) == y_t).float().mean().item()

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'epoch {epoch} train_loss={train_loss:.4f} '
                      f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}',
                      flush=True)

        # 设置阈值
        self.model.eval()
        with torch.no_grad():
            X_t, y_t = to_tensor(X_train, y_train)
            all_probs = self.model(X_t).cpu().numpy()
        pos_probs = all_probs[y_train == 1]
        neg_probs = all_probs[y_train == 0]
        if len(pos_probs) > 0 and len(neg_probs) > 0:
            self.threshold = float((pos_probs.min() + neg_probs.max()) / 2)
            self.threshold = max(0.1, min(0.9, self.threshold))

    def predict_proba(self, feat):
        """feat: (n_frames, 768) numpy → float probability"""
        if self.X_mean is not None and self.X_std is not None:
            feat = (feat - self.X_mean) / self.X_std
        feat_t = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            prob = self.model(feat_t)
        return float(prob.item())

    def save(self, path):
        data = {
            'state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'in_ch': 768,
            'threshold': self.threshold,
            'X_mean': self.X_mean,
            'X_std': self.X_std,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(path, 'rb') as f:
            data = pickle.load(f)
        inst = cls(device=device)
        inst.model.load_state_dict({k: v.to(inst.device) for k, v in data['state_dict'].items()})
        inst.threshold = data['threshold']
        inst.X_mean = data['X_mean']
        inst.X_std = data['X_std']
        return inst
