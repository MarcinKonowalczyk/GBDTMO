#ifndef GBDTMO_TREE_H
#define GBDTMO_TREE_H

#include "mathFunc.h"
#include "datastruct.h"
#include <vector>
#include <map>
#include <iomanip>

struct SplitInfo {
    double gain = -1e16;
    size_t column = 0;
    size_t bin = 0;
    double threshold = 0.0;
    bool is_set = false;

    inline void reset() {
        gain = -1e16, column = 0, bin = 0, threshold = 0.0;
        is_set = false; 
    }

    inline void update(double gain, int column, int bin, double threshold) {
        this->gain = gain;
        this->column = column;
        this->bin = bin;
        this->threshold = threshold;
        this->is_set = true;
    }

    inline void show() { std::cout << "SplitInfo = [\n .gain = " << this->gain << "\n .column = " << this->column << "\n .bin = " << this->bin << "\n .threshold = " << this->threshold << "\n .is_set = " << this->is_set << "\n]\n"; }
};

struct NonLeafNode {
    int parent = 0; // Index of parent
    int left = 0, right = 0; // Left and right node indices
    int column = -1, bin = 0; // Split column and bin index
    double threshold = 0.0; // Split threshold

    NonLeafNode(int p, int c, int b, double t) :
            parent(p), column(c), bin(b), threshold(t) {};
};

struct LeafNode {
    std::vector<double> values;

    LeafNode(double v) : values(1, v) {};
    
    LeafNode(std::vector<double>& v) : values(v) {};
    
    LeafNode(size_t capacity, std::vector<std::pair<double, int>>& v) {
        this->values = std::vector<double>(capacity);
        for (auto& it : v) {
            this->values[it.second] = it.first;
        }
    }
};


struct Tree {
    Tree(bool is_sparse = false) : sparse(is_sparse) {};
    bool sparse;
    int leaf_num = 0, nonleaf_num = 0;
    std::map<int, LeafNode> leafs;
    std::map<int, NonLeafNode> nonleafs;

    inline void clear() {
        leaf_num = 0;
        nonleaf_num = 0;
        leafs.clear();
        nonleafs.clear();
    }

    void _add_nonleaf(const int parent, const int column, const int bin, const double threshold);
    void add_root_nonleaf(const int column, const int bin, const double threshold);
    void add_left_nonleaf(const int parent, const int column, const int bin, const double threshold);
    void add_right_nonleaf(const int parent, const int column, const int bin, const double threshold);
    void shrink(const double learning_rate);
    void _add_leaf(const LeafNode& node);
    void add_left_leaf(const int parent, const LeafNode& node);
    void add_right_leaf(const int parent, const LeafNode& node);

    // predict by original features
    void pred_value_single(
        const double* features,
        double* preds,
        const Shape& shape,
        const size_t n
    ) const;
    void pred_value_multi(
        const double* features,
        double* preds,
        const Shape& shape,
        const size_t n
    ) const;

    // predict by bin maps
    void pred_value_single(
        const uint16_t* features,
        double* preds,
        const Shape& shape,
        const size_t n
    ) const;
    void pred_value_multi(
        const uint16_t* features,
        double* preds,
        const Shape& shape,
        const size_t n
    ) const;

};

#endif /* GBDTMO_TREE_H */
