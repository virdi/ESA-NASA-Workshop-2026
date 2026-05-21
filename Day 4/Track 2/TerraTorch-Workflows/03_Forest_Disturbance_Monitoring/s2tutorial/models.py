"""Torch classifiers for the workshop: linear LR and small MLP.

Device-aware: picks CUDA when available, falls back to CPU. Designed to
behave like sklearn (``fit`` / ``predict`` / ``predict_proba``) so the
audit script can swap backends with a 1-line change.

Single training loop with class-weighted cross-entropy:
``compute_class_weight('balanced', ...)`` reproduces sklearn's
``class_weight='balanced'`` semantics so the two backends are
numerically comparable.

The fit returns *both* train and val macro-F1 — the train-vs-val gap is
the overfit signal the audit reports alongside the val number.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


def _probe_cuda() -> torch.device:
    """Return ``cuda`` only if a tiny CUDA op actually succeeds.

    Whenever we fall back to CPU we also set ``CUDA_VISIBLE_DEVICES=""``
    so later torch internals (e.g. the Adam optimizer's graph-capture
    health check) don't try to touch the broken CUDA context.

    Two reasons for the fallback:
      * ``torch.cuda.is_available()`` returns False — no CUDA driver,
        or the installed wheel was built for a newer CUDA toolkit than
        the driver supports (e.g. torch 2.11 cu130 on a CUDA-12 driver).
      * ``is_available()`` lies — it returns True but the first
        allocation raises ``RuntimeError``.
    """
    import os
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except RuntimeError:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        warnings.warn(
            "torch.cuda.is_available() is True but the first CUDA "
            "allocation failed (driver likely too old). Falling back "
            "to CPU; CUDA hidden for the rest of this process.",
            stacklevel=2,
        )
        return torch.device("cpu")


DEVICE: torch.device = _probe_cuda()


def get_device() -> torch.device:
    """Current device (CUDA if a runtime probe succeeded, else CPU)."""
    return DEVICE


def _is_cpu(device: torch.device) -> bool:
    return device.type == "cpu"


class TorchLR(nn.Module):
    """One-layer linear classifier, parity match for sklearn ``LogisticRegression``.

    Logits = ``X @ W + b``. Pair with class-weighted CE.
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TorchMLP(nn.Module):
    """Two-layer MLP: ``Linear → ReLU → Dropout → Linear``. Modest capacity."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclasses.dataclass
class FitResult:
    train_macro_f1: float
    val_macro_f1: float
    loss_history: list[float]
    epochs: int
    device: str


def _balanced_class_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    full = np.ones(int(classes.max()) + 1, dtype=np.float32)
    for c, wi in zip(classes, w):
        full[int(c)] = float(wi)
    return full


def fit_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int | None = None,
    class_weight: Literal["balanced"] | None = "balanced",
    device: torch.device = DEVICE,
    seed: int = 42,
    verbose: bool = False,
) -> FitResult:
    """Train ``model`` with Adam + class-weighted CE; return train+val F1.

    ``batch_size=None`` runs full-batch Adam (works well for ~14k × 1.5k
    feature matrices on a modern CPU; tiny by GPU standards). Pass an int
    to switch to mini-batch SGD-style updates if you need larger fits.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    has_val = X_val is not None and y_val is not None
    if has_val:
        Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    weight = None
    if class_weight == "balanced":
        w = _balanced_class_weights(np.concatenate(
            [y_train, y_val] if has_val else [y_train]
        ))
        weight = torch.as_tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    n = Xt.shape[0]
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        if batch_size is None:
            optimiser.zero_grad()
            logits = model(Xt)
            loss = criterion(logits, yt)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        else:
            perm = torch.randperm(n, device=device)
            ep_losses = []
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                optimiser.zero_grad()
                logits = model(Xt[idx])
                loss = criterion(logits, yt[idx])
                loss.backward()
                optimiser.step()
                ep_losses.append(float(loss.detach()))
            losses.append(float(np.mean(ep_losses)))
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  ep {epoch + 1:>4d}/{epochs}  loss={losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(Xt).argmax(dim=1).cpu().numpy()
        train_f1 = float(f1_score(y_train, train_pred, average="macro"))
        if has_val:
            val_pred = model(Xv).argmax(dim=1).cpu().numpy()
            val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        else:
            val_f1 = float("nan")

    return FitResult(
        train_macro_f1=train_f1,
        val_macro_f1=val_f1,
        loss_history=losses,
        epochs=epochs,
        device=str(device),
    )


def make_classifier(
    name: str,
    n_features: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """Factory: ``'lr'`` → ``TorchLR``; ``'mlp'`` → ``TorchMLP``."""
    if name == "lr":
        return TorchLR(n_features, n_classes)
    if name == "mlp":
        return TorchMLP(n_features, n_classes, **kwargs)
    raise ValueError(f"Unknown model name {name!r}; expected 'lr' or 'mlp'")


class TorchClassifier:
    """sklearn-shaped wrapper around the fit/predict path.

    Carries a single ``nn.Module``, a fixed device, and the training
    hyper-parameters. Re-instantiate per fold (don't share weights).
    """

    def __init__(
        self,
        name: str,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int | None = None,
        class_weight: Literal["balanced"] | None = "balanced",
        device: torch.device | None = None,
        seed: int = 42,
        verbose: bool = False,
        **model_kwargs,
    ) -> None:
        self.name = name
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.device = device or DEVICE
        self.seed = seed
        self.verbose = verbose
        self.model_kwargs = model_kwargs
        self.model: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self.last_fit: FitResult | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TorchClassifier":
        self.classes_ = np.array(sorted(np.unique(y)))
        n_classes = int(self.classes_.max()) + 1
        # Sanity: skip explicit remapping for now; classes assumed to be
        # contiguous from 0 (true for our binary + 0..3 grouped tasks).
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            warnings.warn(
                f"TorchClassifier expected contiguous class labels 0..K-1; "
                f"got {self.classes_.tolist()}. Predict outputs use raw "
                f"argmax indices.",
                stacklevel=2,
            )
        self.model = make_classifier(
            self.name, n_features=X.shape[1],
            n_classes=n_classes, **self.model_kwargs,
        )
        self.last_fit = fit_classifier(
            self.model, X, y, X_val, y_val,
            epochs=self.epochs, lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            device=self.device, seed=self.seed, verbose=self.verbose,
        )
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        self.model.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self.model(Xt).argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        self.model.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return F.softmax(self.model(Xt), dim=1).cpu().numpy()
