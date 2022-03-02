#include "tree.h"

// predict by original features
// root node is set to -1
void Tree::pred_value_single(const double* features, double* preds, const HyperParameter& hp, const int n) {
    int t_inp = 0, node;
    for (size_t i = 0; i < n; ++i) {
        node = -1;
        do {
            node = features[t_inp + nonleaf[node].column] > nonleaf[node].threshold ? 
                nonleaf[node].right : nonleaf[node].left;
        } while (node < 0);
        preds[i] += int(node != 0) * leaf[node].values[0]; // += if node != 0
        t_inp += hp.inp_dim;
    }
}

void Tree::pred_value_multi(const double* features, double* preds, const HyperParameter& hp, const int n) {
    int t_inp = 0, t_out = 0, node;
    for (size_t i = 0; i < n; ++i) {
        node = -1;
        do {
            node = features[t_inp + nonleaf[node].column] > nonleaf[node].threshold ?
                nonleaf[node].right : nonleaf[node].left;
        } while (node < 0);
        if (node != 0) {
            for (auto &p : leaf[node].values) {
                preds[t_out++] += p;
            }
        } else {
            t_out += hp.out_dim;
        }
        t_inp += hp.inp_dim;
    }
}

// predict by histogram maps
void Tree::pred_value_single(const uint16_t* features, double* preds, const HyperParameter& hp, const int n) {
    for (int i = 0; i < n; ++i) {
        int node = -1;
        while (node < 0) {
            node = features[i + n * nonleaf[node].column] > nonleaf[node].bin
                ? nonleaf[node].right : nonleaf[node].left;
        }
        preds[i] += int(node != 0) * leaf[node].values[0];
    }
}

void Tree::pred_value_multi(const uint16_t* features, double* preds, const HyperParameter& hp, const int n) {
    for (int i = 0; i < n; ++i) {
        int node = -1;
        int t_out = i * hp.out_dim;
        while (node < 0) {
            node = features[i + n * nonleaf[node].column] > nonleaf[node].bin
                ? nonleaf[node].right : nonleaf[node].left;
        }
        if (node != 0) {
            for (auto &p : leaf[node].values) {
                preds[t_out++] += p;
            }
        }
    }
}

