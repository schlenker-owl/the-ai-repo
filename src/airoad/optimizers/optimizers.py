# src/airoad/optimizers/optimizers.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def logistic_loss_grad(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0
) -> tuple[float, np.ndarray, float]:
    """
    Binary logistic with labels y in {0,1}.
    NLL = mean( log(1 + exp(logits)) - y * logits ), where logits = Xw + b
    L2 penalty = l2 * ||w||^2   (bias unpenalized)
    Returns (loss_scalar, grad_w, grad_b)
    """
    # Ensure shapes: X (n,d), y (n,1), w (d,1), b scalar
    n = X.shape[0]
    logits = X @ w + b                      # (n,1)
    p = sigmoid(logits)                     # (n,1)

    # Numerically stable NLL via softplus = logaddexp(0, logits)
    nll = np.mean(np.logaddexp(0.0, logits) - y * logits)   # scalar
    # Robust scalar extraction (avoid float(array([[x]])) deprecation)
    l2_term = float(np.sum(w * w))                           # scalar
    loss = float(nll + l2 * l2_term)

    # Gradients
    grad_w = (X.T @ (p - y)) / n + 2.0 * l2 * w             # (d,1)
    grad_b = float(np.mean(p - y))                           # scalar
    return loss, grad_w, grad_b


@dataclass
class SGD:
    lr: float = 0.1
    def step(self, w: np.ndarray, b: float, grad_w: np.ndarray, grad_b: float):
        return w - self.lr * grad_w, b - self.lr * grad_b


@dataclass
class Momentum:
    lr: float = 0.05
    beta: float = 0.9
    v_w: np.ndarray | None = None
    v_b: float = 0.0
    def step(self, w: np.ndarray, b: float, grad_w: np.ndarray, grad_b: float):
        if self.v_w is None:
            self.v_w = np.zeros_like(w)
        self.v_w = self.beta * self.v_w + (1 - self.beta) * grad_w
        self.v_b = self.beta * self.v_b + (1 - self.beta) * grad_b
        return w - self.lr * self.v_w, b - self.lr * self.v_b


@dataclass
class Adam:
    lr: float = 0.02
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    m_w: np.ndarray | None = None
    v_w: np.ndarray | None = None
    m_b: float = 0.0
    v_b: float = 0.0
    t: int = 0
    def step(self, w: np.ndarray, b: float, grad_w: np.ndarray, grad_b: float):
        if self.m_w is None:
            self.m_w = np.zeros_like(w)
            self.v_w = np.zeros_like(w)
        self.t += 1

        self.m_w = self.b1 * self.m_w + (1 - self.b1) * grad_w
        self.v_w = self.b2 * self.v_w + (1 - self.b2) * (grad_w ** 2)
        m_w_hat = self.m_w / (1 - self.b1 ** self.t)
        v_w_hat = self.v_w / (1 - self.b2 ** self.t)

        self.m_b = self.b1 * self.m_b + (1 - self.b1) * grad_b
        self.v_b = self.b2 * self.v_b + (1 - self.b2) * (grad_b ** 2)
        m_b_hat = self.m_b / (1 - self.b1 ** self.t)
        v_b_hat = self.v_b / (1 - self.b2 ** self.t)

        w_new = w - self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
        b_new = b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
        return w_new, b_new
