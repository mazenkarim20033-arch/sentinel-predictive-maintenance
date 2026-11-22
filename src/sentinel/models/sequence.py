import copy
import io
import time

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import nn

TARGET_SCALE = 125.0


class SeqConfig(BaseModel):
    kind: str = Field("gru", pattern="^(gru|lstm)$")
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 6
    val_asset_fraction: float = 0.15


class SeqNet(nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _inputs(X: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([X, mask[:, :, None].astype(np.float32)], axis=2))


class SequenceRegressor:
    """GRU/LSTM on causal windows. Early stopping uses a held-out slice of the *training assets*."""

    def __init__(self, config: SeqConfig | None = None):
        self.config = config or SeqConfig()
        torch.set_num_threads(2)

    def fit(self, X, mask, y, asset, seed: int = 0) -> "SequenceRegressor":
        cfg = self.config
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        assets = np.unique(asset)
        rng.shuffle(assets)
        val_e = assets[: max(1, round(cfg.val_asset_fraction * len(assets)))]
        is_val = np.isin(asset, val_e)
        inp, tgt = _inputs(X, mask), torch.from_numpy(np.asarray(y, dtype=np.float32) / TARGET_SCALE)
        tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
        self.net_ = SeqNet(cfg.kind, inp.shape[2], cfg.hidden, cfg.layers, cfg.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(seed)
        best, best_state, bad = float("inf"), None, 0
        self.history_: list[dict] = []
        t0 = time.perf_counter()
        for epoch in range(cfg.max_epochs):
            self.net_.train()
            order = torch.from_numpy(tr_idx)[torch.randperm(len(tr_idx), generator=gen)]
            total = 0.0
            for i in range(0, len(order), cfg.batch_size):
                b = order[i : i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(inp[b]), tgt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            val_loss = self._loss(inp[va_idx], tgt[va_idx], loss_fn)
            self.history_.append({"epoch": epoch + 1, "train_loss": total / len(tr_idx), "val_loss": val_loss})
            if val_loss < best - 1e-6:
                best, best_state, bad = val_loss, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin([h["val_loss"] for h in self.history_])) + 1
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def _loss(self, inp, tgt, loss_fn) -> float:
        self.net_.eval()
        with torch.no_grad():
            return float(loss_fn(self.net_(inp), tgt))

    def predict(self, X, mask) -> np.ndarray:
        self.net_.eval()
        inp = _inputs(X, mask)
        with torch.no_grad():
            out = [self.net_(inp[i : i + 2048]) for i in range(0, len(inp), 2048)]
        return torch.cat(out).numpy() * TARGET_SCALE

    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.net_.state_dict(), buf)
        return buf.getvalue()
