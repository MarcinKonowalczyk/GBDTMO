#include "loss.h"

// Mean square error
void mse_grad(const Dataset& data, const int n, const int out_dim, double* g, double* h) {
    auto preds = data.Preds;
    auto labels = data.Label_double;
    int N = n * out_dim;
    for (size_t i = 0; i < N; ++i) {
        g[i] = preds[i] - labels[i];
    }
}

double mse_score(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_double;
    int N = n * out_dim;
    double s = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        s += Sqr(preds[i] - labels[i]);
    }
    return sqrt(s / N);
}

// Binary cross-entropy
void bce_grad(const Dataset& data, const int n, const int out_dim, double* g, double* h) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int N = n * out_dim;
    for (size_t i = 0; i < N; ++i) {
        const double t = 1.0f / (1.0f + exp(-preds[i]));
        g[i] = t - labels[i];
        h[i] = t * (1 - t);
    }
}

double bce_score(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int N = n * out_dim;
    double score = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        double t = log(1 + exp(-preds[i]));
        if (labels[i] == 1) { score += t; }
        else { score += t + preds[i]; }
    }
    return score / N;
}

// Cross-entropy
void ce_grad(const Dataset& data, const int n, const int out_dim, double* g, double* h) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    std::vector<double> rec(out_dim);
    int i, j, idx = 0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < out_dim; ++j) { rec[j] = preds[idx + j]; }
        Softmax(rec);
        for (j = 0; j < out_dim; ++j) {
            g[idx + j] = rec[j];
            h[idx + j] = rec[j] * (1.0f - rec[j]);
        }
        g[idx + labels[i]] -= 1.0f;
        idx += out_dim;
    }
}

double ce_score(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int i, j, idx = 0;
    double score_sum = 0.0f;
    std::vector<double> rec(out_dim);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < out_dim; ++j) { rec[j] = preds[idx + j]; }
        score_sum += Log_sum_exp(rec) - preds[idx + labels[i]];
        idx += out_dim;
    }
    return score_sum;
}

// ----------
void ce_grad_column(const Dataset& data, const int n, const int out_dim, double* g, double* h) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int i, j;
    std::vector<int> idx(out_dim);
    std::vector<double> rec(out_dim);
    for (i = 0; i < n; i += 1) {
        idx[0] = i;
        rec[0] = preds[i];
        for (j = 1; j < out_dim; ++j) {
            idx[j] = idx[j - 1] + n;
            rec[j] = preds[idx[j]];
        }
        Softmax(rec);
        for (j = 0; j < out_dim; ++j) {
            g[idx[j]] = rec[j];
            h[idx[j]] = rec[j] * (1.0f - rec[j]);
        }
        g[idx[labels[i]]] -= 1.0f;
    }
}

double ce_score_column(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int i, j;
    std::vector<int> idx(out_dim);
    std::vector<double> rec(out_dim);
    double score_sum = 0.0f;
    for (i = 0; i < n; i += 1) {
        idx[0] = i;
        rec[0] = preds[i];
        for (j = 1; j < out_dim; ++j) {
            idx[j] = idx[j - 1] + n;
            rec[j] = preds[idx[j]];
        }
        score_sum += Log_sum_exp(rec) - preds[idx[labels[i]]];

    }
    return score_sum;
}

// ----------
double acc_multiclass(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0;
    for (size_t i = 0; i < n; ++i) {
        int idx = i * out_dim;
        double score = preds[idx];
        int ind = 0;
        for (size_t j = 1; j < out_dim; ++j) {
            ++idx;
            if (preds[idx] > score) {
                score = preds[idx];
                ind = j;
            }
        }
        if (ind == labels[i]) { ++acc; }
    }
    return static_cast<double> (acc) / n;
}

double acc_multiclass_column(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0;
    for (size_t i = 0; i < n; ++i) {
        int idx = i;
        double score = preds[idx];
        int ind = 0;
        for (size_t j = 1; j < out_dim; ++j) {
            idx += n;
            if (preds[idx] > score) {
                score = preds[idx];
                ind = j;
            }
        }
        if (ind == labels[i]) { ++acc; }
    }
    return static_cast<double> (acc) / n;
}

double acc_binary(const Dataset& data, const int n, const int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0, N = n * out_dim;
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] == 1) {
            if (preds[i] >= 0.0f) { ++acc; }
        } else {
            if (preds[i] < 0.0f) { ++acc; }
        }
    }
    return static_cast<double> (acc) / N;
}


