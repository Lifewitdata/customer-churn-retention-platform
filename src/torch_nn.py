"""A real PyTorch feedforward neural network, wrapped as an scikit-learn estimator
so it drops into the same Pipeline / ColumnTransformer as the gradient-boosted
models and lands in outputs/model_comparison.csv as its own row.

The wrapper expects already-scaled, already-one-hot input (the shared
ColumnTransformer in features.build_preprocessor does that), does binary
cross-entropy with a class weight to handle the 26.5% churn base rate, and stops
training on an inner stratified validation split rather than a fixed epoch count.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class FeedforwardNet(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = n_features
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchFeedforwardClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible binary classifier backed by torch's autograd."""

    def __init__(self, hidden_layer_sizes=(64, 32), dropout=0.3, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=256, max_epochs=120, patience=12, class_weight=True, val_size=0.2,
                 random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.val_size = val_size
        self.random_state = random_state

    def _seed(self):
        seed = 0 if self.random_state is None else int(self.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32, ensure_2d=True)
        y = y.astype(int)
        self.classes_ = np.unique(y)
        if list(self.classes_) != [0, 1]:
            raise ValueError(f"TorchFeedforwardClassifier needs binary 0/1 labels, got {self.classes_}")
        self.n_features_in_ = X.shape[1]
        self._seed()

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state, stratify=y)
        pos = float(y_tr.sum())
        neg = float(len(y_tr) - pos)
        pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32) if self.class_weight else None

        device = torch.device("cpu")
        net = FeedforwardNet(X.shape[1], tuple(self.hidden_layer_sizes), self.dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        tr = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
        loader = torch.utils.data.DataLoader(tr, batch_size=self.batch_size, shuffle=True,
                                             generator=torch.Generator().manual_seed(int(self.random_state or 0)))
        xv = torch.from_numpy(X_val).to(device)
        yv = torch.from_numpy(y_val.astype(np.float32)).to(device)

        best_val, best_state, bad, history = float("inf"), None, 0, []
        for epoch in range(self.max_epochs):
            net.train()
            run = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(net(xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
                run += float(loss.detach()) * len(xb)
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(xv), yv))
            history.append({"epoch": epoch + 1, "train_loss": run / len(tr), "val_loss": val})
            sched.step(val)
            if val < best_val - 1e-4:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= self.patience:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net.eval()
        self.n_iter_ = len(history)
        self.train_history_ = history
        self.best_val_loss_ = best_val
        return self

    def decision_function(self, X):
        check_is_fitted(self, "net_")
        X = check_array(X, dtype=np.float32, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        with torch.no_grad():
            return self.net_(torch.from_numpy(X)).numpy()

    def predict_proba(self, X):
        logit = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def export_onnx_graph_summary(self) -> list[str]:
        """Layer list, for the model-card / notebook."""
        check_is_fitted(self, "net_")
        return [str(m) for m in self.net_.net if not isinstance(m, nn.Sequential)]
