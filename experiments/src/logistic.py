import numpy as np
import matplotlib.pyplot as plt

# Toy dataset from Figure 1 (Wu et al.), z_i = y_i * x_i
X = np.array([[1, 0.2], [-2, 0.2], [-1, -0.2], [2, -0.2]])
y = np.array([1, 1, -1, -1])
Z = y[:, None] * X


def logistic(z):
    return np.logaddexp(0, -z), 1 / (1 + np.exp(z))


def gd(loss_and_mu, eta, T):
    w, L = np.zeros(2), []
    for _ in range(T):
        l, m = loss_and_mu(Z @ w)
        L.append(l.mean())
        w = w + eta * (m[:, None] * Z).mean(0)  # w - eta * grad L(w)
    return np.array(L)


losses = {
    "logistic": logistic,
}
T = 2**20
t = np.arange(1, T + 1)

# ~4x the area of the previous 16 x 3.5 figure
fig, ax = plt.subplots(figsize=(20, 11))
for name, f in losses.items():
    for k in range(6):
        L = gd(f, 2**k, T)
        L[L < 1e-25] = np.nan
        ax.plot(t, L, label=rf"$\eta = 2^{k}$")
    ax.set(title=name, xlabel="t", ylabel=r"$L(w_t)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(2**-21, 2**5)
ax.legend()

plt.tight_layout()
plt.savefig("./experiments/outputs/fig1.png", dpi=150)
