import numpy as np
import matplotlib.pyplot as plt

# Toy dataset from Figure 1 (Wu et al.), z_i = y_i * x_i
X = np.array([[1, 0.2], [-2, 0.2], [-1, -0.2], [2, -0.2]])
y = np.array([1, 1, -1, -1])
Z = y[:, None] * X


def logistic(z):
    return np.logaddexp(0, -z), 1 / (1 + np.exp(z))


def gd(loss_and_mu, eta, T):
    w, L, W = np.zeros(2), [], []
    for _ in range(T):
        W.append(w)
        l, m = loss_and_mu(Z @ w)
        L.append(l.mean())
        w = w + eta * (m[:, None] * Z).mean(0)  # w - eta * grad L(w)
    return np.array(L), np.array(W)


def sharpness(W):
    """Largest Hessian eigenvalue for each row w of W, shape (T, 2)."""
    s = W @ Z.T  # (T, n)
    c = 1 / (1 + np.exp(-s)) / (1 + np.exp(s))  # l''(s) = sigma(s) sigma(-s)
    a = (c * Z[:, 0] ** 2).mean(1)
    b = (c * Z[:, 0] * Z[:, 1]).mean(1)
    d = (c * Z[:, 1] ** 2).mean(1)
    # closed-form top eigenvalue of the symmetric 2x2 matrix [[a, b], [b, d]]
    return (a + d) / 2 + np.sqrt(((a - d) / 2) ** 2 + b**2)


T = 2**20
t = np.arange(1, T + 1)

# ~4x the area of the previous 16 x 3.5 figure
fig, (ax, ax_s) = plt.subplots(
    2, 1, figsize=(20, 11), sharex=True, height_ratios=[2, 1]
)
for k in range(6):
    L, W = gd(logistic, 2**k, T)
    S = sharpness(W)
    L[L < 1e-25] = np.nan
    S[S < 1e-25] = np.nan
    ax.plot(t, L, label=rf"$\eta = 2^{k}$")
    ax_s.plot(t, S, label=rf"$\eta = 2^{k}$")
ax.set(title="logistic", ylabel=r"$L(w_t)$")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_ylim(2**-21, 2**5)
ax.legend()
ax_s.set(xlabel="t", ylabel=r"$\lambda_{\max}(\nabla^2 L(w_t))$")
ax_s.set_yscale("log", base=2)

plt.tight_layout()
plt.savefig("./experiments/outputs/fig2.png", dpi=150)
