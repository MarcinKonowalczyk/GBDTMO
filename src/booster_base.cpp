#include "booster.h"
#include <limits>
#include <numeric>
#include <algorithm>
#include <random>

//=================================================================
//                                                                 
//  #####     ###     ####  #####                                
//  ##  ##   ## ##   ##     ##                                   
//  #####   ##   ##   ###   #####                                
//  ##  ##  #######     ##  ##                                   
//  #####   ##   ##  ####   #####                                
//                                                                 
//=================================================================

BoosterBase::BoosterBase(const Shape s, HyperParameters p) : shape(s), hp(p) {
    cache = TopkDeque<CacheInfo>(hp.max_caches);
    obj = Objective(hp.loss);
}

//=======================================================================================
//                                                                                       
//   ####  #####  ######  ######  #####  #####     ####                                
//  ##     ##       ##      ##    ##     ##  ##   ##                                   
//   ###   #####    ##      ##    #####  #####     ###                                 
//     ##  ##       ##      ##    ##     ##  ##      ##                                
//  ####   #####    ##      ##    #####  ##   ##  ####                                 
//                                                                                       
//=======================================================================================

void BoosterBase::set_data(float* features, float* preds, size_t n) {
    Data.Features = features;
    Data.preds = preds;
    Data.n = n;
}

void BoosterBase::set_label(float* label) { Data.Label_float = label; }
void BoosterBase::set_label(int32_t* label) { Data.Label_int32 = label; }

//=================================================================
//                                                                 
//   ####    ###    ##       ####                                
//  ##      ## ##   ##      ##                                   
//  ##     ##   ##  ##      ##                                   
//  ##     #######  ##      ##                                   
//   ####  ##   ##  ######   ####                                
//                                                                 
//=================================================================

void BoosterBase::calc_maps() {
    // Calculate Data.Maps and bins
    std::vector<std::vector<float>> bins;

    Data.train_maps.resize(shape.inp_dim);
    for (size_t i = 0; i < shape.inp_dim; ++i) {
        // Get features column
        std::vector<float> features_column;
        features_column.reserve(Data.n);
        for (size_t j = 0; j < Data.n; ++j) { features_column.push_back(Data.Features[i + j*shape.inp_dim]); }

        // Construct bins for the column
        std::vector<float> bins_column;
        construct_bin_column(features_column, bins_column, hp.max_bins);
        bins.push_back(bins_column);

        // Map features in the particular column
        // NOTE: This means that maps are column-major
        Data.train_maps[i].resize(Data.n);
        map_bin_column(features_column, Data.train_maps[i], bins_column);
    }

    // Number of bins in each column
    bin_nums.clear();
    bin_values.clear();
    float float_max = std::numeric_limits<float>::max();
    for (auto& bin : bins) {
        bin_nums.push_back(bin.size() + 1);
        auto tmp = bin;
        tmp.push_back(float_max);
        bin_values.push_back(tmp);
    }
}

void BoosterBase::calc_eval_fraction() {
    Data.train_order.resize(Data.n);
    std::iota(Data.train_order.begin(), Data.train_order.end(), 0);
    Data.eval_order.clear();

    if (hp.eval_fraction >= 0.0) {
        size_t n_eval = static_cast<size_t>(hp.eval_fraction * Data.n);

        std::vector<size_t> perm(Data.n);
        std::iota(perm.begin(), perm.end(), 0);
        std::random_shuffle(perm.begin(), perm.end());

        // Trim order to the train_subset
        // TODO: Could use just std::partition, no?
        //       I don't think the order matters, except for, maybe, the cache access.
        auto it = std::stable_partition(
            Data.train_order.begin(), Data.train_order.end(),
            [&perm, n_eval](size_t o) { return perm[o] >= n_eval; }
        );
        Data.eval_order.assign(it, Data.train_order.end());
        Data.train_order.erase(it, Data.train_order.end());
    }
}

void BoosterBase::rebuild_order(
    const std::vector<size_t>& order,
    std::vector<size_t>& order_l,
    std::vector<size_t>& order_r,
    const size_t split_column,
    const uint16_t bin
) const {
    const auto map_column = Data.train_maps[split_column];
    int count_l = 0, count_r = 0;
    for (size_t o : order) {
        if (map_column[o] <= bin) {
            // assert(count_l < order_l.size());
            order_l[count_l++] = o;
        } else {
            // assert(count_r < order_r.size());
            order_r[count_r++] = o;
        }
    }
}


float* BoosterBase::malloc_G() const {
    return (float*) malloc(Data.n * shape.out_dim * sizeof(float));
}

float* BoosterBase::malloc_H(const bool constHessian = true, const float constValue = 0.0) const {
    const size_t n = Data.n * shape.out_dim;
    float* H = (float*) malloc(n * sizeof(float));
    if (constHessian) std::fill_n(H, n, constValue);
    return H;
}

void BoosterBase::reset() {
    cache.clear();
    tree.clear();
    trees.clear();
    // TODO: test? should this be Data.n * shape.out_dim???
    //       also, when is this actually used? what's the purpose here?
    //       retrain? then should have a way to set the parameters(!)
    // std::fill_n(Data.Preds, Data.n * shape.out_dim, 0.0);
}

//===============================================
//                                               
//  ##   #####                                 
//  ##  ##   ##                                
//  ##  ##   ##                                
//  ##  ##   ##                                
//  ##   #####                                 
//                                               
//===============================================

void BoosterBase::dump_nonleaf_sizes(uint16_t* nonleaf_sizes) const {
    for (size_t i = 0; i < trees.size(); ++i) {
        nonleaf_sizes[i] = trees[i].nonleafs.size();
    }
}

void BoosterBase::dump_leaf_sizes(uint16_t* leaf_sizes) const {
    for (size_t i = 0; i < trees.size(); ++i) {
        leaf_sizes[i] = trees[i].leafs.size();
    }
}

void BoosterBase::dump_nonleaf_nodes(int* trees, float* thresholds) const {
    int i = 0, j = 0;
    for(auto& tree : this->trees) {
        for (auto& it : tree.nonleafs) {
            auto node = it.second;
            trees[i++] = it.first;
            trees[i++] = node.parent;
            trees[i++] = node.left;
            trees[i++] = node.right;
            trees[i++] = node.column;
            thresholds[j++] = node.threshold;
        }
    }
}

void BoosterBase::dump_leaf_nodes(float* leaves) const {
    int k = 0;
    for(auto& tree : trees ) {
        for (auto& leaf : tree.leafs) {
            auto node = leaf.second;
            for (auto& value : node.values) {
                leaves[k++] = value;
            }
        }
    }
}
