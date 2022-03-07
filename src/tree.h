#ifndef GBDTMO_TREE_H
#define GBDTMO_TREE_H

#include "mathFunc.h"
#include "datastruct.h"
#include <vector>
#include <map>
#include <iomanip>

struct SplitInfo {
    double gain = -1e8;
    int column = -1;
    int bin = -1;
    double threshold = 0.0f;

    inline void reset() { gain = -1e8, column = -1; }

    inline void update(double gain, int column, int bin, double threshold) {
        this->gain = gain;
        this->column = column;
        this->bin = bin;
        this->threshold = threshold;
    }
};

struct NonLeafNode {
    int parent = 0, left = 0, right = 0;
    int column = -1;
    int bin = 0;
    double threshold = 0.0f;

    NonLeafNode() {};

    NonLeafNode(int parent_, int column_, int bin_, double threshold_) :
            parent(parent_), column(column_), bin(bin_), threshold(threshold_) {};
};

struct LeafNode {
    LeafNode(int n = 1) : values(n, 0) {};
    int parent;
    std::vector<double> values;

    inline void Update(int parent, double value) {
        this->parent = parent;
        this->values[0] = value;
    }

    inline void Update(int parent, std::vector<double>& values) {
        this->parent = parent;
        this->values.assign(values.begin(), values.end());
    }

    inline void Update(int parent, std::vector<std::pair<double, int>>& values) {
        this->parent = parent;
        for (auto &it : values) {
            this->values[it.second] = it.first;
        }
    }
};


struct Tree {
    Tree(bool is_sparse = false) : sparse(is_sparse) {};
    bool sparse;
    int leaf_num = 0, nonleaf_num = 0;
    std::map<int, LeafNode> leaf;
    std::map<int, NonLeafNode> nonleaf;

    inline void clear() {
        leaf_num = 0;
        nonleaf_num = 0;
        leaf.clear();
        nonleaf.clear();
    }

    inline void add_leaf(const LeafNode& node, bool left) {
        ++leaf_num;
        leaf.emplace(leaf_num, node);
        if (left) {
            nonleaf[node.parent].left = leaf_num;
        } else {
            nonleaf[node.parent].right = leaf_num;
        }
    }

    inline void add_nonleaf(const NonLeafNode& node, bool left) {
        --nonleaf_num;
        nonleaf.emplace(nonleaf_num, node);
        if (left) {
            nonleaf[node.parent].left = nonleaf_num;
        } else {
            nonleaf[node.parent].right = nonleaf_num;
        }
    }

    inline void shrinkage(double lr) {
        for (int i = 1; i < leaf.size() + 1; ++i) {
            for (auto &p : leaf[i].values) {
                p *= lr;
            }
        }
    }

    // predict by original features
    void pred_value_single(const double* features, double* preds, const HyperParameters& hp, const int n);
    void pred_value_multi(const double* features, double* preds, const HyperParameters& hp, const int n);

    // predict by bin maps
    void pred_value_single(const uint16_t* features, double* preds, const HyperParameters& hp, const int n);
    void pred_value_multi(const uint16_t* features, double* preds, const HyperParameters& hp, const int n);

};

#endif /* GBDTMO_TREE_H */
