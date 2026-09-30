"""Joint classifier and stable marginal likelihood for partially observed labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

CLASS_BITS = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class Config:
    width: int = 128
    dropout: float = .3
    lr: float = .001
    weight_decay: float = .01
    epochs: int = 150
    batch_size: int = 64


class JointNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2049, config.width), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.width, 4))

    def forward(self, x):
        return self.net(x)


def partial_nll(logits: torch.Tensor, labels: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Each row contributes once, regardless of whether one or two tests are known.

    Hidden label values are ignored, including NaN and sentinel -1. Observed labels
    must be binary. The mask never grants access to hidden measurements.
    """
    if logits.ndim != 2 or logits.shape[1] != 4 or labels.shape != (len(logits), 2) or observed.shape != labels.shape:
        raise ValueError("Expected logits [N,4], labels and observation mask [N,2]")
    observed = observed.bool()
    if not torch.all((labels[observed] == 0) | (labels[observed] == 1)):
        raise ValueError("Observed labels must be 0 or 1")
    active = observed.any(dim=1)
    if not active.any():
        return logits.sum() * 0
    bits = torch.tensor(CLASS_BITS, device=logits.device)
    compatible = ((labels[:, None, :] == bits[None, :, :]) | ~observed[:, None, :]).all(dim=2)
    logp = torch.log_softmax(logits[active], dim=1)
    return -torch.logsumexp(logp.masked_fill(~compatible[active], -torch.inf), dim=1).mean()


def normalized_x(x, mean, std):
    result = np.asarray(x, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 2049 or not np.isfinite(result).all():
        raise ValueError("Expected finite [N,2049] features")
    result[:, -1] = (result[:, -1] - mean) / std
    return result


class Member:
    def __init__(self, config, network, potency_mean, potency_std, epochs):
        self.config, self.network = config, network
        self.potency_mean, self.potency_std, self.epochs = potency_mean, potency_std, epochs

    def predict(self, x):
        self.network.eval()
        with torch.no_grad():
            xt = torch.from_numpy(normalized_x(x, self.potency_mean, self.potency_std))
            return torch.softmax(self.network(xt), dim=1).numpy()


def fit_member(x, labels, observed, config, seed, validation=None):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    active = np.asarray(observed).any(axis=1)
    if not active.any():
        raise ValueError("Training requires at least one observed outcome")
    # Completely unobserved records cannot alter preprocessing or minibatches.
    x, labels, observed = np.asarray(x)[active], np.asarray(labels)[active], np.asarray(observed)[active]
    mean, std = float(x[:, -1].mean()), max(float(x[:, -1].std()), 1e-6)
    xt = torch.from_numpy(normalized_x(x, mean, std))
    yt, mt = torch.as_tensor(labels), torch.as_tensor(observed, dtype=torch.bool)
    net = JointNetwork(config)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if validation is not None:
        vx, vy = validation
        vxt = torch.from_numpy(normalized_x(vx, mean, std))
        vyt = torch.as_tensor(vy[:, 0] * 2 + vy[:, 1], dtype=torch.long)
    best_loss, best_state, best_epoch, stale = float("inf"), None, config.epochs, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        net.train()
        order = rng.permutation(len(x))
        accumulated = 0.
        for start in range(0, len(x), config.batch_size):
            idx = order[start:start+config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = partial_nll(net(xt[idx]), yt[idx], mt[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            accumulated += loss.item() * len(idx)
        entry = {"epoch": epoch, "train_nll": accumulated / len(x)}
        if validation is not None:
            net.eval()
            with torch.no_grad():
                val_loss = nn.functional.cross_entropy(net(vxt), vyt).item()
            entry["validation_nll"] = val_loss
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
        history.append(entry)
        if validation is not None and stale >= 25:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return Member(config, net, mean, std, best_epoch), history


def temperature_scale(probabilities, temperature):
    return softmax(np.log(np.clip(probabilities, 1e-8, 1)) / temperature, axis=1)


def fit_temperature(probabilities, labels):
    labels = np.asarray(labels, dtype=int)
    def objective(log_t):
        p = temperature_scale(probabilities, np.exp(log_t))
        return -np.log(np.clip(p[np.arange(len(p)), labels], 1e-8, 1)).mean()
    result = minimize_scalar(objective, bounds=(np.log(.5), np.log(5.)), method="bounded")
    return float(np.exp(result.x))


class Ensemble:
    def __init__(self, members, temperature=1.):
        self.members, self.temperature = members, temperature

    def raw_predict(self, x):
        return np.mean([m.predict(x) for m in self.members], axis=0)

    def predict(self, x):
        return temperature_scale(self.raw_predict(x), self.temperature)

    def calibrate(self, x, y):
        self.temperature = fit_temperature(self.raw_predict(x), y)

    def save(self, path: Path, metadata=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "class_order": list(CLASS_BITS), "temperature": self.temperature, "metadata": metadata or {}, "members": [{"config": asdict(m.config), "state_dict": m.network.state_dict(), "potency_mean": m.potency_mean, "potency_std": m.potency_std, "epochs": m.epochs} for m in self.members]}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or payload["class_order"] != list(CLASS_BITS):
            raise ValueError("Unsupported checkpoint schema")
        members = []
        for record in payload["members"]:
            config = Config(**record["config"])
            network = JointNetwork(config)
            network.load_state_dict(record["state_dict"])
            members.append(Member(config, network, record["potency_mean"], record["potency_std"], record["epochs"]))
        return cls(members, payload["temperature"])


class IndependentBaseline:
    def __init__(self, kind="rf", strength=4, seed=0):
        self.kind, self.strength, self.seed = kind, strength, seed
        self.models, self.temperature = [], 1.

    def fit(self, x, labels, observed):
        self.mean, self.std = float(x[:, -1].mean()), max(float(x[:, -1].std()), 1e-6)
        xn = normalized_x(x, self.mean, self.std)
        self.models = []
        for k in range(2):
            use = observed[:, k].astype(bool)
            classes = np.unique(labels[use, k])
            if not len(classes):
                self.models.append(.5)
                continue
            if len(classes) == 1:
                self.models.append(float(classes[0]))
                continue
            if self.kind == "rf":
                model = RandomForestClassifier(n_estimators=300, min_samples_leaf=int(self.strength), max_features="sqrt", n_jobs=2, random_state=self.seed+k)
            else:
                model = LogisticRegression(C=float(self.strength), max_iter=1500, solver="lbfgs")
            model.fit(xn[use], labels[use, k])
            self.models.append(model)
        return self

    def raw_predict(self, x):
        xn = normalized_x(x, self.mean, self.std)
        ps = [np.full(len(x), m) if isinstance(m, float) else m.predict_proba(xn)[:, list(m.classes_).index(1)] for m in self.models]
        d, h = ps
        return np.column_stack([(1-d)*(1-h), (1-d)*h, d*(1-h), d*h])

    def predict(self, x):
        return temperature_scale(self.raw_predict(x), self.temperature)

    def calibrate(self, x, y):
        self.temperature = fit_temperature(self.raw_predict(x), y)


def conditional_probabilities(probabilities, dd2=None, hepg2=None):
    """Remaining endpoint pass chances after a known outcome; no hidden inputs."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Expected normalized four-class probabilities")
    if dd2 is not None and hepg2 is not None:
        raise ValueError("Supply only one observed test")
    if dd2 is not None:
        if dd2 not in (0, 1):
            raise ValueError("Dd2 observation must be 0 or 1")
        denominator = p[:, 2*dd2] + p[:, 2*dd2+1]
        return np.divide(p[:, 2*dd2+1], denominator, out=np.full(len(p), np.nan), where=denominator>0)
    if hepg2 is not None:
        if hepg2 not in (0, 1):
            raise ValueError("HepG2 observation must be 0 or 1")
        denominator = p[:, hepg2] + p[:, hepg2+2]
        return np.divide(p[:, hepg2+2], denominator, out=np.full(len(p), np.nan), where=denominator>0)
    raise ValueError("Supply an observed endpoint")
