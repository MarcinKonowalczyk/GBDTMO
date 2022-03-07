#include "histogram.h"

void histogram_single(std::vector<int32_t>& order, Histogram& Hist, uint16_t* maps, double* G, double* H) {
    for (auto i : order) {
        size_t bin = maps[i];
        ++Hist.count[bin];
        Hist.g[bin] += G[i];
        Hist.h[bin] += H[i];
    }
    // integration
    for (int i = 1; i < Hist.count.size(); ++i) {
        Hist.count[i] += Hist.count[i - 1];
        Hist.g[i] += Hist.g[i - 1];
        Hist.h[i] += Hist.h[i - 1];
    }
}

void histogram_multi(std::vector<int32_t>& order, Histogram& Hist, uint16_t* maps, double* G, double* H, int out_dim) {
    for (int32_t i : order) {
        ++Hist.count[maps[i]];
        size_t bin = maps[i] * out_dim;
        size_t ind = i * out_dim;
        for (size_t j = 0; j < out_dim; ++j) {
            Hist.g[bin+j] += G[ind+j];
            Hist.h[bin+j] += H[ind+j];
        }
    }
    // integration
    size_t ind = 0;
    for (size_t i = 1; i < Hist.count.size(); ++i) {
        Hist.count[i] += Hist.count[i - 1];
        for (size_t j = 0; j < out_dim; ++j) {
            Hist.g[ind + out_dim] += Hist.g[ind];
            Hist.h[ind + out_dim] += Hist.h[ind];
            ++ind;
        }
    }
}

