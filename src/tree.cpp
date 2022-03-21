#include "tree.h"


[[gnu::pure]]
inline int Tree::traverse_tree(const double* const feature) const {
    int node_index = -1;
    do {
        auto node = nonleafs.at(node_index);
        node_index = feature[node.column] > node.threshold ? node.right : node.left;
    } while (node_index < 0);
    return node_index;
}


// predict by original features
// root node is set to -1
void Tree::pred_value_single(
    const double* const features,
    double* const preds,
    const Shape& shape,
    const size_t n
) const {
    if (nonleafs.size() == 0) return; // Early out for empty trees
    for (size_t i = 0; i < n; ++i) {
        auto feature = features + (i * shape.inp_dim);
        int leaf_index = traverse_tree(feature);
        // int node_index = -1;
        // do {
            // auto node = nonleafs.at(node_index);
            // node_index = feature[node.column] > node.threshold ? node.right : node.left;
        // } while (node_index < 0);
        if (leaf_index != 0) preds[i] += leafs.at(leaf_index).values[0];
    }
}

void Tree::pred_value_multi(
    const double* const features,
    double* const preds,
    const Shape& shape,
    const size_t n
) const {
    if (nonleafs.size() == 0) return; // Early out for empty trees
    for (size_t i = 0; i < n; ++i) {
        auto feature = features + (i * shape.inp_dim);
        int leaf_index = traverse_tree(feature);
        // int node_index = -1;
        // do {
        //     auto node = nonleafs.at(node_index);
        //     node_index = feature[node.column] > node.threshold ? node.right : node.left;
        // } while (node_index < 0);
        if (leaf_index != 0) {
            size_t t_out = i * shape.out_dim;
            for (double p : leafs.at(leaf_index).values) { preds[t_out++] += p; }
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
    if (nonleafs.find(parent) == nonleafs.end()) {
        std::cout << "Tree::add_left_nonleaf " << parent << "\n";
        std::cout << "nonleafs = \n";
        for (auto it = nonleafs.begin(); it != nonleafs.end(); it++) {
            std::cout << it->first << " , " << it->second.parent << " " << it->second.left << " " << it->second.right << "\n";
        }
        std::cout << "\n";
        throw;
    }
    nonleafs.at(parent).left = nonleaf_num;
};

void Tree::add_right_nonleaf(const int parent, const int column, const int bin, const double threshold) {
    _add_nonleaf(parent, column, bin, threshold);
    if (nonleafs.find(parent) == nonleafs.end()) {
        std::cout << "Tree::add_right_nonleaf " << parent << "\n";
        std::cout << "nonleafs = \n";
        for (auto it = nonleafs.begin(); it != nonleafs.end(); it++) {
            std::cout << it->first << " , " << it->second.parent << " " << it->second.left << " " << it->second.right << "\n";
        }
        throw;
    }
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
    if (nonleafs.find(parent) == nonleafs.end()) {
        std::cout << "Tree::add_left_leaf " << parent << "\n";
        std::cout << "nonleafs (" << nonleafs.size() <<") = \n";
        for (auto it = nonleafs.begin(); it != nonleafs.end(); it++) {
            std::cout << it->first << " , " << it->second.parent << " " << it->second.left << " " << it->second.right << "\n";
        }
        throw;
    }
    nonleafs.at(parent).left = leaf_num;
}

void Tree::add_right_leaf(const int parent, const LeafNode& node) {
    _add_leaf(node);
    if (nonleafs.find(parent) == nonleafs.end()) {
        std::cout << "Tree::add_right_leaf " << parent << "\n";
        std::cout << "nonleafs (" << nonleafs.size() <<") = \n";
        for (auto it = nonleafs.begin(); it != nonleafs.end(); it++) {
            std::cout << it->first << " , " << it->second.parent << " " << it->second.left << " " << it->second.right << "\n";
        }
        throw;
    }
    nonleafs.at(parent).right = leaf_num;
}