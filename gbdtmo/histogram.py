import numpy as np

def _construct_bin_column(X: np.array, max_bins: int) -> np.array:
    X, count = np.unique(X, return_counts=True)
    sum_count = count.sum()

    if len(X) == 1:
        bins = np.array([], 'float64')

    elif len(X) == 2:
        bins = np.array([(X[0] * count[0] + X[1] * count[1]) / sum_count], dtype=np.float64)

    elif len(X) <= max_bins:
        bins = np.zeros(len(X) - 1, 'float64')
        for i in range(len(X) - 1):
            bins[i] = (X[i] + X[i + 1]) / 2.0

    elif len(X) > max_bins:
        count = np.cumsum(count)
        t, p = 0, len(X) / float(max_bins)
        bins = np.zeros(max_bins - 1, 'float64')
        for i in range(len(X)):
            if count[i] >= p:
                bins[t] = X[i]
                t += 1
                p = count[i] + (sum_count - count[i]) / float(max_bins - t)
            if t == max_bins - 1: break

    return bins


def _map_bin_column(X, bins):
    bins = np.insert(bins, 0, -np.inf)
    bins = np.insert(bins, len(bins), np.inf)
    return np.searchsorted(bins, X, side='left').astype('uint16') - 1


def get_bins_maps(X: np.array, max_bins: int) -> tuple[list, np.array]:
    bins = [_construct_bin_column(x, max_bins) for x in X.T]
    maps = [_map_bin_column(x, b) for x, b in zip(X.T, bins)]

    # NOTE: don't convert bins to array since they can have uneven lengths
    return bins, np.stack(maps, axis=1)


if __name__ == '__main__':
    X = np.random.rand(10000, 10)
    bins, maps = get_bins_maps(X, 8)
    print(bins[0])
