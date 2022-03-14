#include "tree.h"

// predict by original features
// root node is set to -1
void Tree::pred_value_single(
    const double* features,
    double* preds,
    const HyperParameters& hp,
    const int n
) const {
    int t_inp = 0, node_index;
    for (size_t i = 0; i < n; ++i) {
        node_index = -1;
        do {
            auto node = nonleaf.at(node_index);
            node_index = features[t_inp + node.column] > node.threshold ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) preds[i] += leaf.at(node_index).values[0];
        t_inp += hp.inp_dim;
    }
}

void Tree::pred_value_multi(
    const double* features,
    double* preds,
    const HyperParameters& hp,
    const int n
) const {
    int t_inp = 0, node_index;
    for (size_t i = 0; i < n; ++i) {
        node_index = -1;
        do {
            auto node = nonleaf.at(node_index);
            node_index = features[t_inp + node.column] > node.threshold ? node.right : node.left;
        } while (node_index < 0);

        if (node_index != 0) {
            int t_out = i * hp.out_dim;
            for (double p : leaf.at(node_index).values) { preds[t_out++] += p; }
        }
        t_inp += hp.inp_dim;
    }
}

// predict by histogram maps
void Tree::pred_value_single(
    const uint16_t* features,
    double* preds,
    const HyperParameters& hp,
    const int n
) const {
    for (int i = 0; i < n; ++i) {
        int node_index = -1;
        do {
            auto node = nonleaf.at(node_index);
            node_index = features[i + n * node.column] > node.bin ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) preds[i] += leaf.at(node_index).values[0];
    }
}

void Tree::pred_value_multi(
    const uint16_t* features,
    double* preds,
    const HyperParameters& hp,
    const int n
) const {
    for (int i = 0; i < n; ++i) {
        int node_index = -1;
        do {
            auto node = nonleaf.at(node_index);
            node_index = features[i + n * node.column] > node.bin ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) {
            int t_out = i * hp.out_dim;
            for (double p : leaf.at(node_index).values) { preds[t_out++] += p; }
        }
    }
}

