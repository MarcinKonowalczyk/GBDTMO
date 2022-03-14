#ifndef GBDTMO_TREE_H
#define GBDTMO_TREE_H

#include "mathFunc.h"
#include "datastruct.h"
#include <vector>
#include <map>
#include <iomanip>

struct SplitInfo {
    double gain = -1e8;
    size_t column = 0;
    size_t bin = 0;
    double threshold = 0.0f;
    bool isset = false;

    inline void reset() {
        gain = -1e16, column = 0, bin = 0, threshold = 0.0f;
        isset = false; 
    }

    inline void update(double gain, int column, int bin, double threshold) {
        this->gain = gain;
        this->column = column;
        this->bin = bin;
        this->threshold = threshold;
        this->isset = true;
    }

    inline void show() { std::cout << "SplitInfo = [\n .gain = " << this->gain << "\n .column = " << this->column << "\n .bin = " << this->bin << "\n .threshold = " << this->threshold << "\n .isset = " << this->isset << "\n]\n"; }
};

struct NonLeafNode {
    int parent = 0, left = 0, right = 0;
    int column = -1;
    int bin = 0;
    double threshold = 0.0f;

    NonLeafNode() {};

    NonLeafNode(int p, int c, int b, double t) :
            parent(p), column(c), bin(b), threshold(t) {};
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
        // for (size_t i = 1; i < leaf.size() + 1; ++i) {
        //     for (auto &p : leaf[i].values) {
        //         p *= lr;
        //     }
        // }
        for (auto l : leaf) {
            for (auto& p : l.second.values) {
                p *= lr;
            }
        }
    }

    // predict by original features
    void pred_value_single(
        const double* features,
        double* preds,
        const HyperParameters& hp,
        const int n
    ) const;
    void pred_value_multi(
        const double* features,
        double* preds,
        const HyperParameters& hp,
        const int n
    ) const;

    // predict by bin maps
    void pred_value_single(
        const uint16_t* features,
        double* preds,
        const HyperParameters& hp,
        const int n
    ) const;
    void pred_value_multi(
        const uint16_t* features,
        double* preds,
        const HyperParameters& hp,
        const int n
    ) const;

};

#endif /* GBDTMO_TREE_H */
