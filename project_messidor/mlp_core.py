"""
From-scratch NumPy MLP for binary classification, with:
  - standard mini-batch SGD/Adam training
  - DP-SGD training (per-sample gradient clipping + Gaussian noise), Abadi et al. 2016 style
Used identically by the centralized, FedAvg, and FedAvg+DP stages so that all three
pipelines share EXACTLY the same model class / preprocessing / evaluation code.

NOTE ON ENVIRONMENT: PyTorch / Opacus / Flower are not installable in this sandbox
(no network access). This module reproduces the same DP-SGD mechanism Opacus implements
(per-sample grad clip -> sum -> add N(0, sigma^2 * C^2 I) -> average) and FedAvg exactly
as it would be implemented with those libraries. It is a drop-in-shaped replacement.
"""
import numpy as np

RNG_SEED = 42


def set_seed(seed=RNG_SEED):
    np.random.seed(seed)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NumpyMLP:
    """2-hidden-layer MLP: in -> h1 (ReLU) -> h2 (ReLU) -> 1 (sigmoid), BCE loss."""

    def __init__(self, n_in, h1=16, h2=8, seed=RNG_SEED):
        rng = np.random.RandomState(seed)
        # He init
        self.W1 = rng.randn(n_in, h1) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(h1)
        self.W2 = rng.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(1)
        self.shapes = [self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape, self.W3.shape, self.b3.shape]
        self.m = None
        self.v = None
        self.t = 0

    # ---------- param <-> flat vector helpers (needed for FedAvg averaging & DP noise) ----------
    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [p.copy() for p in params]

    def clone(self):
        m = NumpyMLP.__new__(NumpyMLP)
        m.shapes = self.shapes
        m.set_params(self.get_params())
        m.m = None
        m.v = None
        m.t = 0
        return m

    def n_params(self):
        return sum(p.size for p in self.get_params())

    # ---------- forward / backward ----------
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3
        out = sigmoid(z3).ravel()
        cache = (X, z1, a1, z2, a2, z3)
        return out, cache

    def predict_proba(self, X):
        p, _ = self.forward(X)
        return p

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)

    def loss(self, X, y, l2=0.0):
        p, _ = self.forward(X)
        eps = 1e-9
        bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if l2 > 0:
            bce += l2 * sum(np.sum(p_ ** 2) for p_ in [self.W1, self.W2, self.W3])
        return bce

    def backward_batch(self, X, y, l2=0.0):
        """Standard batch-averaged gradient (fast path, non-DP)."""
        n = X.shape[0]
        p, (Xc, z1, a1, z2, a2, z3) = self.forward(X)
        dz3 = (p - y).reshape(-1, 1) / n
        dW3 = a2.T @ dz3 + 2 * l2 * self.W3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 + 2 * l2 * self.W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = Xc.T @ dz1 + 2 * l2 * self.W1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def per_sample_gradients(self, X, y):
        """Returns a list (len=n_samples) of flattened per-sample gradient vectors.
        Needed for correct DP-SGD (Abadi et al. 2016): clip each sample's gradient
        individually BEFORE aggregating, not the batch-averaged gradient."""
        n = X.shape[0]
        p, (Xc, z1, a1, z2, a2, z3) = self.forward(X)
        dz3 = (p - y).reshape(-1, 1)               # (n,1)
        dW3 = a2[:, :, None] * dz3[:, None, :]      # (n,h2,1)
        db3 = dz3                                   # (n,1)
        da2 = dz3 @ self.W3.T                        # (n,h2)
        dz2 = da2 * (z2 > 0)                          # (n,h2)
        dW2 = a1[:, :, None] * dz2[:, None, :]       # (n,h1,h2)
        db2 = dz2                                    # (n,h2)
        da1 = dz2 @ self.W2.T                         # (n,h1)
        dz1 = da1 * (z1 > 0)                           # (n,h1)
        dW1 = Xc[:, :, None] * dz1[:, None, :]        # (n,in,h1)
        db1 = dz1                                     # (n,h1)

        flat = np.concatenate([
            dW1.reshape(n, -1), db1.reshape(n, -1),
            dW2.reshape(n, -1), db2.reshape(n, -1),
            dW3.reshape(n, -1), db3.reshape(n, -1),
        ], axis=1)
        return flat  # (n, n_params)

    def unflatten(self, flat_vec):
        out, i = [], 0
        for shp in self.shapes:
            size = int(np.prod(shp))
            out.append(flat_vec[i:i + size].reshape(shp))
            i += size
        return out

    def apply_flat_update(self, flat_grad, lr, optimizer='adam', beta1=0.9, beta2=0.999, eps=1e-8):
        if optimizer == 'adam':
            if self.m is None:
                self.m = np.zeros_like(flat_grad)
                self.v = np.zeros_like(flat_grad)
                self.t = 0
            self.t += 1
            self.m = beta1 * self.m + (1 - beta1) * flat_grad
            self.v = beta2 * self.v + (1 - beta2) * (flat_grad ** 2)
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            step = lr * flat_grad

        grads = self.unflatten(step)
        self.W1 -= grads[0]; self.b1 -= grads[1]
        self.W2 -= grads[2]; self.b2 -= grads[3]
        self.W3 -= grads[4]; self.b3 -= grads[5]

    # ---------- training loops ----------
    def train_standard(self, X, y, epochs=1, batch_size=32, lr=0.05, l2=1e-4, seed=0, shuffle=True, optimizer='adam'):
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        for ep in range(epochs):
            idx = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                grads = self.backward_batch(X[b], y[b], l2=l2)
                flat_grad = np.concatenate([g.flatten() for g in grads])
                self.apply_flat_update(flat_grad, lr, optimizer=optimizer)
        return self

    def train_dp_sgd(self, X, y, epochs, batch_size, lr, max_grad_norm, noise_multiplier, seed=0, optimizer='adam'):
        """Abadi et al. (2016) DP-SGD: per-sample clip to C, sum, add N(0, (sigma*C)^2 I), divide by batch size."""
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        n_steps = 0
        for ep in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                Xb, yb = X[b], y[b]
                bsz = Xb.shape[0]
                if bsz == 0:
                    continue
                per_sample = self.per_sample_gradients(Xb, yb)   # (bsz, P)
                norms = np.linalg.norm(per_sample, axis=1, keepdims=True)
                clip_factor = np.minimum(1.0, max_grad_norm / (norms + 1e-12))
                clipped = per_sample * clip_factor
                summed = clipped.sum(axis=0)
                noise = rng.normal(0, noise_multiplier * max_grad_norm, size=summed.shape)
                noised_avg = (summed + noise) / bsz
                self.apply_flat_update(noised_avg, lr, optimizer=optimizer)
                n_steps += 1
        return n_steps
