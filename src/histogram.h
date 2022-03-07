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

void histogram_single(std::vector<int32_t> &, Histogram &, uint16_t *, double *, double *);
void histogram_multi(std::vector<int32_t> &, Histogram &, uint16_t *, double *, double *, int);

#endif /* GBDTMO_HISTOGRAM_H */