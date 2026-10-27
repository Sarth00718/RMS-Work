"""
Renyi Differential Privacy (RDP) accountant for the sampled Gaussian mechanism,
following Mironov, Talwar & Zhang (2019) "Renyi DP of the Sampled Gaussian Mechanism" —
the same mathematical mechanism used internally by Opacus / TF-Privacy.

Implemented from the published formulas (not copied from any library source), since
Opacus itself is unavailable in this offline sandbox.

RDP of sampled Gaussian mechanism at integer order alpha, sampling rate q, noise
multiplier sigma:
    RDP(alpha) = 1/(alpha-1) * log( sum_{k=0}^{alpha} C(alpha,k) (1-q)^(alpha-k) q^k
                                     * exp(k(k-1) / (2*sigma^2)) )
Composition over T steps: RDP_total(alpha) = T * RDP(alpha)
Conversion to (epsilon, delta):
    epsilon = min_alpha [ RDP_total(alpha) + log(1/delta) / (alpha - 1) ]
"""
import numpy as np
from scipy.special import comb, logsumexp


def _rdp_sampled_gaussian(q, sigma, alpha):
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * sigma ** 2)
    log_terms = []
    for k in range(0, alpha + 1):
        log_c = np.log(comb(alpha, k, exact=False) + 1e-300)
        log_term = log_c + (alpha - k) * np.log(1 - q + 1e-300) + k * np.log(q + 1e-300) \
            + (k * (k - 1)) / (2.0 * sigma ** 2)
        log_terms.append(log_term)
    return logsumexp(log_terms) / (alpha - 1)


def compute_epsilon(n_samples, batch_size, noise_multiplier, epochs, delta, sigma_floor=1e-6):
    """Returns (epsilon, best_alpha). q = sampling rate per step, T = total steps."""
    if noise_multiplier <= sigma_floor:
        return float('inf'), None
    q = batch_size / n_samples
    steps_per_epoch = max(1, int(np.ceil(n_samples / batch_size)))
    T = steps_per_epoch * epochs
    orders = list(range(2, 64)) + [70, 80, 100, 128, 150, 200, 256]
    best_eps, best_alpha = float('inf'), None
    for alpha in orders:
        rdp = _rdp_sampled_gaussian(q, noise_multiplier, alpha) * T
        eps = rdp + np.log(1.0 / delta) / (alpha - 1)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha
    return best_eps, best_alpha


def compute_epsilon_federated(n_samples_per_client, batch_size, noise_multiplier,
                               local_epochs, rounds, delta):
    """Composition across communication rounds: each round each client does `local_epochs`
    passes of DP-SGD; the client's own dataset is what it subsamples from each round."""
    eps, alpha = compute_epsilon(
        n_samples=n_samples_per_client,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=local_epochs * rounds,
        delta=delta,
    )
    return eps, alpha
