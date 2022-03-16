#ifndef GBDTMO_HISTOGRAM_H
#define GBDTMO_HISTOGRAM_H

#include <vector>

struct Histogram {
    Histogram(size_t n = 1, size_t d = 1)
        : size(n), count(n, 0), g(n * d, 0), h(n * d, 0) {}

    size_t size;
    std::vector<int> count;
    std::vector<double> g;
    std::vector<double> h;

    inline void operator-(const Histogram& x) {

        for (size_t i = 0; i < x.count.size(); ++i) {
            count[i] -= x.count[i];
        }
        for (size_t i = 0; i < x.g.size(); ++i) {
            g[i] -= x.g[i];
            h[i] -= x.h[i];
        }
    }
};

// void histogram_single(std::vector<size_t>& order, Histogram& Hit, uint16_t*, double*, double*);
// void histogram_multi(std::vector<size_t>& order, Histogram&, uint16_t*, double*, double*, int);

void construct_bin_column(
    std::vector<double> features_column, // Note: don't pass by reference. Copy. 'feature' is not modified outside of the function.
    std::vector<double>& bins_column,
    const uint16_t max_bins
);

// Calculate binning of a single column of the features matrix
void map_bin_column(
    const std::vector<double> features_column,
    std::vector<uint16_t>& map_column,
    std::vector<double>& bins
);

// void calculate_histogram_maps(
//     const double* features,
//     uint16_t* maps,
//     std::vector<std::vector<double>>& bins,
//     const size_t n,
//     const size_t inp_dim,
//     const uint16_t max_bins
// );

#endif /* GBDTMO_HISTOGRAM_H */