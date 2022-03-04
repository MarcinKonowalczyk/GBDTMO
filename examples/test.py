import argparse
import numpy as np

import os, sys

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from gbdtmo import GBDTMulti, load_lib, GBDTSingle

parser = argparse.ArgumentParser()
parser.add_argument("-lr", default=0.1, type=float)
parser.add_argument("-depth", default=2, type=int)
args = parser.parse_args()

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


# def regression():
#     inp_dim, out_dim = 10, 5
#     booster = GBDTMulti(LIB, out_dim=out_dim, params=dict(max_depth=args.depth, lr=args.lr, loss="mse"))
#     seed = 42
#     X_train, X_test = np.random.rand(10000, inp_dim), np.random.rand(100, inp_dim)
#     y_train, y_test = f(X_train, out_dim, seed), f(X_test, out_dim, seed)

#     booster.set_data((X_train, y_train), (X_test, y_test))
#     booster.train(1000)
#     # booster.dump(b"regression.txt")
#     yp = booster.predict(X_test)

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
    inp_dim, out_dim = 10, 2
    params = params = dict(max_depth=args.depth, lr=args.lr, loss="mse", early_stop=10)

    seed = 42
    with seed_rng(seed):
        X_train, X_test = np.random.rand(10000, inp_dim), np.random.rand(100, inp_dim)
        M = np.random.randn(5 * inp_dim, out_dim)

    # Function to approximate with the gradient booster
    f = lambda X: np.apply_along_axis(lambda a: a - np.mean(a), 0, np.c_[X, X**2, X**(1 / 2), X**3, X**(1 / 3)] @ M)
    y_train, y_test = f(X_train), f(X_test)

    booster = GBDTSingle(LIB, out_dim=y_train.shape[1], params=params)
    # booster = GBDTMulti(LIB, out_dim=y_train.shape[1], params=params)
    booster.set_data((X_train, y_train), (X_test, y_test))
    booster.train(10000)
    yp = booster.predict(X_test)
    y = y_test

    eval_score = np.sqrt(np.mean((y[:, 0] - yp[:, 0])**2))
    print(f"first column eval_score = {eval_score}")

    if y.shape[1] == 2:
        cla()
        subplot(2, 1, 1)
        I = np.argsort(y[:, 0])
        plot(y[I, 0])
        plot(yp[I, 0])
        grid("on")
        subplot(2, 1, 2)
        I = np.argsort(y[:, 1])
        plot(y[I, 1])
        plot(yp[I, 1])
        grid("on")
    else:
        cla()
        I = np.argsort(y[:, 0])
        plot(y[I, 0])
        plot(yp[I, 0])
        grid("on")
