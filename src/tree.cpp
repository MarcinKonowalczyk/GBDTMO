#include "tree.h"

// predict by original features
// root node is set to -1
void Tree::pred_value_single(
    const double* features,
    double* preds,
    const HyperParameters& hp,
    const size_t n
) const {
    size_t t_inp = 0;
    for (size_t i = 0; i < n; ++i) {
        int node_index = -1;
        do {
            auto node = nonleafs.at(node_index);
            node_index = features[t_inp + node.column] > node.threshold ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) preds[i] += leafs.at(node_index).values[0];
        t_inp += hp.inp_dim;
    }
}

void Tree::pred_value_multi(
    const double* features,
    double* preds,
    const HyperParameters& hp,
    const size_t n
) const {
    size_t t_inp = 0;
    for (size_t i = 0; i < n; ++i) {
        int node_index = -1;
        do {
            auto node = nonleafs.at(node_index);
            node_index = features[t_inp + node.column] > node.threshold ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) {
            int t_out = i * hp.out_dim;
            for (double p : leafs.at(node_index).values) { preds[t_out++] += p; }
        }
        t_inp += hp.inp_dim;
    }
}

// predict by histogram maps
void Tree::pred_value_single(
    const uint16_t* features,
    double* preds,
    const HyperParameters& hp,
    const size_t n
) const {
    for (int i = 0; i < n; ++i) {
        int node_index = -1;
        do {
            auto node = nonleafs.at(node_index);
            node_index = features[i + n * node.column] > node.bin ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) preds[i] += leafs.at(node_index).values[0];
    }
}

void Tree::pred_value_multi(
    const uint16_t* features,
    double* preds,
    const HyperParameters& hp,
    const size_t n
) const {
    for (int i = 0; i < n; ++i) {
        int node_index = -1;
        do {
            auto node = nonleafs.at(node_index);
            node_index = features[i + n * node.column] > node.bin ? node.right : node.left;
        } while (node_index < 0);
        if (node_index != 0) {
            int t_out = i * hp.out_dim;
            for (double p : leafs.at(node_index).values) { preds[t_out++] += p; }
        }
    }
}

void Tree::_add_nonleaf(const int parent, const int column, const int bin, const double threshold) {
    nonleafs.emplace(--nonleaf_num, NonLeafNode(parent, column, bin, threshold));
};

void Tree::add_root_nonleaf(const int column, const int bin, const double threshold) {
    _add_nonleaf(0, column, bin, threshold); // Root node has null parent
};

void Tree::add_left_nonleaf(const int parent, const int column, const int bin, const double threshold) {
    _add_nonleaf(parent, column, bin, threshold);
    nonleafs.at(parent).left = nonleaf_num;
};

void Tree::add_right_nonleaf(const int parent, const int column, const int bin, const double threshold) {
    _add_nonleaf(parent, column, bin, threshold);
    nonleafs.at(parent).right = nonleaf_num;
};

void Tree::shrink(const double learning_rate) {
    for (auto& leaf : leafs) {
        for (auto& value : leaf.second.values) {
            value *= learning_rate;
        }
    }
};

void Tree::_add_leaf(const LeafNode& node) {
    leafs.emplace(++leaf_num, node);
}

void Tree::add_left_leaf(const int parent, const LeafNode& node) {
    _add_leaf(node);
    nonleafs.at(parent).left = leaf_num;
}

void Tree::add_right_leaf(const int parent, const LeafNode& node) {
    _add_leaf(node);
    nonleafs.at(parent).right = leaf_num;
}