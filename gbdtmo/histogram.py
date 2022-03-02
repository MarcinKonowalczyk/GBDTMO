import numpy as np

# from numba import jit


# @jit(forceobj=True)
def construct_bin_column(x: np.array, max_bins: int) -> np.array:
    x, count = np.unique(x, return_counts=True)
    sum_count = np.sum(count)
    if len(x) == 1:
        bins = np.array([], 'float64')

    elif len(x) == 2:
        bins = np.array([(x[0] * count[0] + x[1] * count[1]) / sum_count], dtype=np.float64)

    elif len(x) <= max_bins:
        bins = np.zeros(len(x) - 1, 'float64')
        for i in range(len(x) - 1):
            bins[i] = (x[i] + x[i + 1]) / 2.0

    elif len(x) > max_bins:
        count = np.cumsum(count)
        t, p = 0, len(x) / float(max_bins)
        bins = np.zeros(max_bins - 1, 'float64')
        for i in range(len(x)):
            if count[i] >= p:
                bins[t] = x[i]
                t += 1
                p = count[i] + (sum_count - count[i]) / float(max_bins - t)
            if t == max_bins - 1: break

    return bins


def map_bin_column(x, bins):
    bins = np.insert(bins, 0, -np.inf)
    bins = np.insert(bins, len(bins), np.inf)

    return np.searchsorted(bins, x, side='left').astype('uint16') - 1


def _get_bins_maps(x_column: np.array, max_bins: int) -> tuple:
    bins = construct_bin_column(x_column, max_bins)
    maps = map_bin_column(x_column, bins)

    return bins, maps


def get_bins_maps(x: np.array, max_bins: int) -> tuple[list, np.array]:
    out = []
    for i in range(x.shape[-1]):
        out.append(_get_bins_maps(x[:, i], max_bins))

    bins, maps = [], []
    while out:
        _bin, _map = out.pop(0)
        bins.append(_bin)
        maps.append(_map)
    return bins, np.stack(maps, axis=1)


if __name__ == '__main__':
    x = np.random.rand(10000, 10)
    bins, maps = get_bins_maps(x, 8, 2)
    print(bins[0])
