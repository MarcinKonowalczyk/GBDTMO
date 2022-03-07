import argparse
import numpy as np

import os, sys

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from gbdtmo import GBDTMulti, load_lib, GBDTSingle

# parser = argparse.ArgumentParser()
# parser.add_argument("-lr", default=0.1, type=float)
# parser.add_argument("-depth", default=2, type=int)
# args = parser.parse_args()

LIB = load_lib("../build/gbdtmo.so" if os.path.exists("../build/gbdtmo.so") else "./build/gbdtmo.so")

from contextlib import contextmanager


@contextmanager
def seed_rng(random_state):
    """Temporarilly seed the state of the random number generator"""
    if random_state:
        old_state = np.random.get_state()
        np.random.seed(random_state)
    try:
        yield
    finally:
        if random_state:
            np.random.set_state(old_state)


# Generate short "unique" string to be put int the corner of the figure, to easily, visually, make sure the figure got regenerated
from random import sample
from string import ascii_letters

uid = lambda: ''.join(sample(ascii_letters, 10))

# TODO: test classification too

# def classification():
#     inp_dim, out_dim = 10, 5
#     params = dict(max_depth=args.depth, lr=args.lr, loss="ce")
#     booster = GBDTMulti(LIB, out_dim=out_dim, params=params)
#     X_train = np.random.rand(10000, inp_dim)
#     y_train = np.random.randint(0, out_dim, size=(10000, )).astype("int32")
#     X_valid = np.random.rand(10000, inp_dim)
#     y_valid = np.random.randint(0, out_dim, size=(10000, )).astype("int32")
#     booster.set_data((X_train, y_train), (X_valid, y_valid))
#     booster.train(20)
#     booster.dump(b"classification.txt")

if __name__ == '__main__':
    booster_shape = (10, 2)
    seed = 42

    booster_params = dict(max_depth=2, lr=0.1, loss="mse", early_stop=50, verbose=False, seed=seed)
    with seed_rng(seed):
        X_train, X_test = np.random.rand(10000, booster_shape[0]), np.random.rand(100, booster_shape[0])
        M = np.random.randn(5 * booster_shape[0], booster_shape[1])

    # Function to approximate with the gradient booster
    f = lambda X: np.apply_along_axis(lambda a: a - np.mean(a), 0, np.c_[X, X**2, X**(1 / 2), X**3, X**(1 / 3)] @ M)
    y_train, y_test = f(X_train), f(X_test)

    booster_single = GBDTSingle(LIB, shape=booster_shape, params=booster_params)
    booster_multi = GBDTMulti(LIB, shape=booster_shape, params=booster_params)

    booster_single.set_data((X_train, y_train), (X_test, y_test))
    booster_multi.set_data((X_train, y_train), (X_test, y_test))

    booster_single.train(100)
    booster_multi.train(100)

    yp_single = booster_single.predict(X_test)
    yp_multi = booster_multi.predict(X_test)
    y = y_test

    # Plot
    import matplotlib
    import matplotlib.pyplot as plt

    cm = [p['color'] for p in matplotlib.rcParams['axes.prop_cycle']]

    y_params = dict(color='k', alpha=0.3, label='y')
    yp_single_params = dict(marker='o', linewidth=0, markersize=2, color=cm[0], label='yp_single')
    yp_multi_params = dict(marker='x', linewidth=0, markersize=2, color=cm[1], label='yp_multi')

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    I = np.argsort(y[:, 0])
    ax1.plot(y[I, 0], **y_params)
    ax1.plot(yp_single[I, 0], **yp_single_params)
    ax1.plot(yp_multi[I, 0], **yp_multi_params)
    ax1.grid("on")
    ax1.legend()
    ax1.set_title("GBDT Single/Multi y vs yp")

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    I = np.argsort(y[:, 1])
    ax2.plot(y[I, 1], **y_params)
    ax2.plot(yp_single[I, 1], **yp_single_params)
    ax2.plot(yp_multi[I, 1], **yp_multi_params)
    ax2.grid("on")
    ax2.set_xlabel("(sorted) sample number")

    plt.setp(ax1.get_xticklabels(), visible=False)

    text_params = dict(ha='right', va='bottom', transform=fig.transFigure, size=8, alpha=0.3)
    plt.text(0.99, 0.01, f"{uid()}", **text_params)

    # plt.show()
    plt.savefig("test.png")
