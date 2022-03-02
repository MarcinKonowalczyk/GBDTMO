#include "tree.h"

// predict by original features
// root node is set to -1
void Tree::pred_value_single_(double *features, double *preds, HyperParameter &hp, int n) {
    int t_inp = 0, node;
    for (size_t i = 0; i < n; ++i) {
        node = -1;
        while (node < 0) {
            if (features[t_inp + nonleaf[node].column] > nonleaf[node].threshold) {
                node = nonleaf[node].right;
            } else {
                node = nonleaf[node].left;
            }
        }
        if (node != 0) { preds[i] += leaf[node].values[0]; }
        t_inp += hp.inp_dim;
    }
}

void Tree::pred_value_multi_(double *features, double *preds, HyperParameter &hp, int n) {
    int t_inp = 0, t_out = 0, node;
    for (size_t i = 0; i < n; ++i) {
        node = -1;
        while (node < 0) {
            if (features[t_inp + nonleaf[node].column] > nonleaf[node].threshold) {
                node = nonleaf[node].right;
            } else {
                node = nonleaf[node].left;
            }
        }
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

void Tree::pred_value_single(double *features, double *preds, HyperParameter &hp, int n) {
    pred_value_single_(features, preds, hp, n);
}

void Tree::pred_value_multi(double *features, double *preds, HyperParameter &hp, int n) {
    pred_value_multi_(features, preds, hp, n);
}

// predict by histogram maps
void Tree::pred_value_single(uint16_t *features, double *preds, HyperParameter &hp, int n) {
    for (int i = 0; i < n; ++i) {
        int node = -1;
        while (node < 0) {
            if (features[i + n * nonleaf[node].column] > nonleaf[node].bin) {
                node = nonleaf[node].right;
            } else {
                node = nonleaf[node].left;
            }
        }
        if (node != 0) { preds[i] += leaf[node].values[0]; }
    }
}

void Tree::pred_value_multi(uint16_t *features, double *preds, HyperParameter &hp, int n) {
    for (int i = 0; i < n; ++i) {
        int node = -1;
        int t_out = i * hp.out_dim;
        while (node < 0) {
            if (features[i + n * nonleaf[node].column] > nonleaf[node].bin) {
                node = nonleaf[node].right;
            } else {
                node = nonleaf[node].left;
            }
        }
        if (node != 0) {
            for (auto &p : leaf[node].values) {
                preds[t_out++] += p;
            }
        }
    }
}

