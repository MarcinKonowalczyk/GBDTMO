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

def f(X, out_dim, seed = None):
    """Function to approximate by the GBDT"""
    inp_dim = X.shape[1]
    with seed_rng(seed):
        M = np.random.randn(5*inp_dim, out_dim)
    y = np.c_[X,X**2,X**(1/2),X**3,X**(1/3)] @ M
    y = y - np.mean(y, axis=0)
    return y


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
    params = params=dict(max_depth=args.depth, lr=args.lr, loss="mse")
    
    seed = 42
    X_train, X_test = np.random.rand(10000, inp_dim), np.random.rand(100, inp_dim)
    y_train, y_test = f(X_train, out_dim, seed), f(X_test, out_dim, seed)

    booster_single = GBDTSingle(LIB, out_dim=out_dim, params=params)
    booster_single.set_data((X_train, y_train), (X_test, y_test))
    booster_single.train(31)
    yp = booster_single.predict(X_test)
    y = y_test

    cla(); plot(y[:,0]); plot(yp[:,0]);
# if __name__ == '__main__':
#     inp_dim, out_dim = 10, 5
#     params = params=dict(max_depth=args.depth, lr=args.lr, loss="mse")
    
#     seed = 42
#     X_train, X_test = np.random.rand(10000, inp_dim), np.random.rand(100, inp_dim)
#     y_train, y_test = f(X_train, out_dim, seed), f(X_test, out_dim, seed)

#     booster_mutli = GBDTMulti(LIB, out_dim=out_dim, params=params)
#     booster_mutli.set_data((X_train, y_train), (X_test, y_test))
#     booster_mutli.train(100)
#     yp = booster_mutli.predict(X_test)
#     y = y_test
