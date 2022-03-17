#include "loss.h"

//============================================================
//                                                            
//  ###    ###   ####  #####                                
//  ## #  # ##  ##     ##                                   
//  ##  ##  ##   ###   #####                                
//  ##      ##     ##  ##                                   
//  ##      ##  ####   #####                                
//                                                            
//============================================================
// mean square error

void mse_grad(const Dataset& data, const size_t out_dim, double* const g, double* const h) { 
    const size_t N = data.n * out_dim;
    for (size_t o = 0; o < N; ++o) { g[o] = data.preds[o] - data.Label_double[o]; }
}

double mse_score(const Dataset& data, const size_t out_dim, const double* const g) {
    double s = 0.0;
    const size_t N = data.n * out_dim;
    for (size_t o = 0; o < N; ++o) { s += Sqr(g[o]); }
    return sqrt(s / N);
}

double mse_partial_score(const Dataset& data, const size_t out_dim, const double* const g, const bool score_train) {
    auto order = score_train ? data.train_order : data.eval_order;
    double s = 0.0;
    for (size_t o : order) {
        const size_t row_offset = o*out_dim;
        for (size_t i = 0; i < out_dim; ++i) {
            s += Sqr(g[row_offset+i]);
        }
    }
    return sqrt(s / (order.size() * out_dim));
}

//========================================================
//                                                        
//  #####    ####  #####                                
//  ##  ##  ##     ##                                   
//  #####   ##     #####                                
//  ##  ##  ##     ##                                   
//  #####    ####  #####                                
//                                                        
//========================================================
// binary cross-entropy

void bce_grad(const Dataset& data, const size_t out_dim, double* const g, double* const h) {
    // TODO: Is there a perf difference betweek this and size_t k = 0;
    //       and k++ in the loop? Look into (using compiler explorer?)
    const size_t N = data.n * out_dim;
    for (size_t o = 0; o < N; ++o) {
        const double t = 1.0 / (1.0 + exp(-data.preds[o]));
        g[o] = t - data.Label_int32[o];
        h[o] = t * (1 - t);
    }
}

double bce_score(const Dataset& data, const size_t out_dim, const double* const g) {
    // TODO: Use gradient ??
    double score = 0.0;
    const size_t N = data.n * out_dim;
    for (size_t o = 0; o < N; ++o) {
        const double p = data.preds[o];
        const double t = log(1 + exp(-p));
        score += (data.Label_int32[o] == 1) ? t : t + p;
    }
    return score / N;
}

double bce_partial_score(const Dataset& data, const size_t out_dim, const double* const g, const bool score_train) {
    auto order = score_train ? data.train_order : data.eval_order;
    double score = 0.0;
    for (size_t o : order) {
        const size_t row_offset = o*out_dim;
        for (size_t i = 0; i < out_dim; ++i) {
            const double p = data.preds[row_offset + i];
            const double t = log(1 + exp(-p));
            score += (data.Label_int32[row_offset + i] == 1) ? t : t + p;
        }
    }
    return score / (order.size() * out_dim);
}

//================================================
//                                                
//   ####  #####                                
//  ##     ##                                   
//  ##     #####                                
//  ##     ##                                   
//   ####  #####                                
//                                                
//================================================

// Cross-entropy
// TODO: Precompute rec??
void ce_grad(const Dataset& data, const size_t out_dim, double* const g, double* const h) {
    std::vector<double> rec(out_dim);
    size_t idx = 0;
    for (size_t i = 0; i < data.n; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            rec[j] = data.preds[idx + j];
        }
        Softmax(rec);
        for (size_t j = 0; j < out_dim; ++j) {
            g[idx + j] = rec[j];
            h[idx + j] = rec[j] * (1.0 - rec[j]);
        }
        g[idx + data.Label_int32[i]] -= 1.0;
        idx += out_dim;
    }
}

double ce_score(const Dataset& data, const size_t out_dim, const double* const g) {
    size_t idx = 0;
    double score = 0.0;
    std::vector<double> rec(out_dim);
    for (size_t i = 0; i < data.n; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            rec[j] = data.preds[idx + j];
        }
        score += Log_sum_exp(rec) - data.preds[idx + data.Label_int32[i]];
        idx += out_dim;
    }
    return score;
}

//==============================================================================================================
//                                                                                                              
//   ####  #####         ####   #####   ##      ##   ##  ###    ###  ##     ##                                
//  ##     ##           ##     ##   ##  ##      ##   ##  ## #  # ##  ####   ##                                
//  ##     #####        ##     ##   ##  ##      ##   ##  ##  ##  ##  ##  ## ##                                
//  ##     ##           ##     ##   ##  ##      ##   ##  ##      ##  ##    ###                                
//   ####  #####         ####   #####   ######   #####   ##      ##  ##     ##                                
//                                                                                                              
//==============================================================================================================

void ce_column_grad(const Dataset& data, const size_t out_dim, double* const g, double* const h) {
    const size_t n = data.n;
    const auto preds = data.preds;
    const auto labels = data.Label_int32;
    int i, j;
    std::vector<size_t> idx(out_dim);
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
            const size_t _idx = idx[j];
            g[_idx] = rec[j];
            h[_idx] = rec[j] * (1.0 - rec[j]);
        }
        g[idx[labels[i]]] -= 1.0;
    }
}

double ce_column_score(const Dataset& data, const size_t out_dim, const double* const g) {
    const size_t n = data.n;
    const auto preds = data.preds;
    const auto labels = data.Label_int32;
    int i, j;
    std::vector<int> idx(out_dim);
    std::vector<double> rec(out_dim);
    double score_sum = 0.0;
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

//=========================================================
//                                                         
//    ###     ####   ####                                
//   ## ##   ##     ##                                   
//  ##   ##  ##     ##                                   
//  #######  ##     ##                                   
//  ##   ##   ####   ####                                
//                                                         
//=========================================================

double acc_multiclass(const Dataset& data, const size_t out_dim, const double* const g) {
    const size_t n = data.n;
    const auto preds = data.preds;
    const auto labels = data.Label_int32;
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

double acc_multiclass_column(const Dataset& data, const size_t out_dim, const double* const g) {
    const size_t n = data.n;
    const auto preds = data.preds;
    const auto labels = data.Label_int32;
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

// nary(const Dataset& data, const size_t out_dim, const double* const g, const bool score_train) {
//     auto order = score_train ? data.train_order : data.eval_order;
//     const size_t n = data.n;
//     const auto preds = data.preds;
//     const auto labels = data.Label_int32;
//     size_t N = data.n * out_dim;
//     size_t acc = 0;
//     for (size_t i = 0; i < N; ++i) {
//         if (labels[i] == 1) {
//             if (preds[i] >= 0.0) { ++acc; }
//         } else {
//             if (preds[i] < 0.0) { ++acc; }
//         }
//     }
//     return static_cast<double> (acc) / N;
// }

// Accuracy score for binary labels
double acc_binary(const Dataset& data, const size_t out_dim, const double* const g, const bool score_train) {
    auto order = score_train ? data.train_order : data.eval_order;
    // const size_t N = data.n * out_dim;
    size_t acc = 0;
    for (size_t o : order) {
        const size_t row_offset = o*out_dim;
        for (size_t i = 0; i < out_dim; ++i) {
            // TODO: This can be written as XNAND, i think.
            if (data.Label_int32[row_offset+i] == 1) {
                if (data.preds[row_offset+i] >= 0.0) { ++acc; }
            } else {
                if (data.preds[row_offset+i] < 0.0) { ++acc; }
            }
        }
    }
    return static_cast<double> (acc) / (order.size() * out_dim);
}