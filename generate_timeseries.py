import numpy as np


def randc(beta, N):
    X = np.random.randn(N)
    if beta == 0: return X

    m1 = np.mean(X)
    s1 = np.sqrt(np.mean((X - m1)**2))

    X = np.fft.rfft(X)
    k = np.fft.rfftfreq(N)
    k = 1 / np.where(k == 0, float('inf'), k) if beta < 0 else k
    k = k**np.abs(beta)
    # k = k / np.sqrt(np.mean(k**2))
    X = np.fft.irfft(X * k)

    m2 = np.mean(X)
    s2 = np.sqrt(np.mean((X - m2)**2))

    s12 = s1 / s2
    X *= s12
    X += (m1 - m2 * s12)

    return X


def generate(N):
    x = np.zeros(N, dtype=np.double)
    t = np.arange(N)
    for _ in range(10):
        a = (0.8 * np.random.rand() + 0.2)
        f = 30 * (0.5 * np.random.rand() + 0.5)
        p = np.random.rand() * 2 * np.pi
        x += np.sin(t * 2 * np.pi / N * f + p)

    # Add autoregressive component
    N_kernel = 10
    K = np.r_[np.random.random(N_kernel), 1, np.zeros(N_kernel)]
    x = np.convolve(np.pad(x, N_kernel, mode='edge'), K, mode='valid')
    x *= 1 / np.std(x)

    # Add coloured noise

    x += 1.0 * randc(-2, N)
    x += 0.5 * randc(-0.5, N)

    return x


if __name__ == "__main__":
    d, hh = 30, 48  # days, half-hours in a day
    N = d * hh  # Number of datapoints
    t, x = np.arange(N), generate(N)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, x)
    ax1.set_ylim((-5, 5))
    ax1.grid('on')
    ax1.set_title("entire timeseries")
    ax1.set_xticks(np.arange(0, d + 1, 3) * hh)
    ax1.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t[:3 * hh], x[:3 * hh], '.-')
    ax2.set_ylim((-5, 5))
    ax2.grid('on')
    ax2.xaxis.set_ticklabels([])
    ax2.set_xticks(np.arange(4) * hh)
    ax2.set_title("first 3 \"days\"")

    # plt.savefig("generate_timeseries.png")
