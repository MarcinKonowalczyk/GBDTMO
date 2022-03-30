#ifndef GBDTMO_TREE_H
#define GBDTMO_TREE_H

#include "mathFunc.h"
#include "datastruct.h"
#include <vector>
#include <map>
#include <iomanip>

struct SplitInfo {
    float gain = -1e16;
    size_t column = 0;
    size_t bin = 0;
    float threshold = 0.0;
    bool is_set = false;

    inline void reset() {
        gain = -1e16, column = 0, bin = 0, threshold = 0.0;
        is_set = false; 
    }

    inline void update(float gain, int column, int bin, float threshold) {
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
    float threshold = 0.0; // Split threshold

    NonLeafNode(int p, int c, int b, float t) :
            parent(p), column(c), bin(b), threshold(t) {};
};

struct LeafNode {
    std::vector<float> values;

    LeafNode(float v) : values(1, v) {};
    
    LeafNode(std::vector<float>& v) : values(v) {};
    
    LeafNode(size_t capacity, std::vector<std::pair<float, int>>& v) {
        this->values = std::vector<float>(capacity);
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

    void _add_nonleaf(const int parent, const int column, const int bin, const float threshold);
    void add_root_nonleaf(const int column, const int bin, const float threshold);
    void add_left_nonleaf(const int parent, const int column, const int bin, const float threshold);
    void add_right_nonleaf(const int parent, const int column, const int bin, const float threshold);
    void shrink(const float learning_rate);
    void _add_leaf(const LeafNode& node);
    void add_left_leaf(const int parent, const LeafNode& node);
    void add_right_leaf(const int parent, const LeafNode& node);

    // predict
    inline int traverse_tree(const float* const feature) const;

    void pred_value_single(
        const float* const features,
        float* const preds,
        const Shape& shape,
        const size_t n
    ) const;
    void pred_value_multi(
        const float* const features,
        float* const preds,
        const Shape& shape,
        const size_t n
    ) const;
};

#endif /* GBDTMO_TREE_H */
