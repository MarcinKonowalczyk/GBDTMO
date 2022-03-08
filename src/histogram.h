#ifndef GBDTMO_HISTOGRAM_H
#define GBDTMO_HISTOGRAM_H

#include <vector>

struct Histogram {
    Histogram(int n = 1, int d = 1) : count(n, 0), g(n * d, 0), h(n * d, 0) {}

    std::vector<int> count;
    std::vector<double> g;
    std::vector<double> h;

    inline void operator-(const Histogram &x) {

        for (int i = 0; i < x.count.size(); ++i) {
            count[i] -= x.count[i];
        }
        for (int i = 0; i < x.g.size(); ++i) {
            g[i] -= x.g[i];
            h[i] -= x.h[i];
        }
    }
};

void histogram_single(
    std::vector<size_t>& order,
    Histogram& Hist,
    uint16_t* maps,
    double* G,
    double* H
);

void histogram_multi(
    std::vector<size_t>& order,
    Histogram& Hist,
    uint16_t* maps,
    double* G,
    double* H,
    int out_dim
);

// void histogram_single(std::vector<size_t>& order, Histogram& Hit, uint16_t*, double*, double*);
// void histogram_multi(std::vector<size_t>& order, Histogram&, uint16_t*, double*, double*, int);

void calculate_histogram_maps(
    const double* features,
    uint16_t* maps,
    std::vector<std::vector<double>>& bins,
    const size_t n,
    const size_t inp_dim,
    const uint16_t max_bins
);

#endif /* GBDTMO_HISTOGRAM_H */