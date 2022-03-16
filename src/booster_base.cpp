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

BoosterBase::BoosterBase(HyperParameters p) : hp(p) {
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

void BoosterBase::set_data(double* features, double* preds, size_t n) {
    Data.Features = features;
    Data.preds = preds;
    Data.n = n;
}

void BoosterBase::set_label(double* label) { Data.Label_double = label; }
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
    std::vector<std::vector<double>> bins;

    Data.train_maps.resize(hp.inp_dim);
    for (size_t i = 0; i < hp.inp_dim; ++i) {
        // Get features column
        std::vector<double> features_column;
        features_column.reserve(Data.n);
        for (size_t j = 0; j < Data.n; ++j) { features_column.push_back(Data.Features[i + j*hp.inp_dim]); }

        // Construct bins for the column
        std::vector<double> bins_column;
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
    double double_max = std::numeric_limits<double>::max();
    for (auto& bin : bins) {
        bin_nums.push_back(bin.size() + 1);
        auto tmp = bin;
        tmp.push_back(double_max);
        bin_values.push_back(tmp);
    }
}

void BoosterBase::calc_eval_fraction() {
    Data.train_order.resize(Data.n);
    std::iota(Data.train_order.begin(), Data.train_order.end(), 0);

    if (hp.eval_fraction >= 0.0) {
        size_t n_eval = static_cast<size_t>(hp.eval_fraction * Data.n);

        // TODO: this can be done so much better!!
        std::vector<size_t> perm(Data.n);
        std::iota(perm.begin(), perm.end(), 0);

        std::random_shuffle(perm.begin(), perm.end());

        // Trim order to the train_subset
        Data.train_order.erase(std::remove_if(
            Data.train_order.begin(), Data.train_order.end(),
            [&perm, n_eval](size_t o) { return perm[o] < n_eval; }
        ), Data.train_order.end());
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

double* BoosterBase::malloc_G() const {
    return (double*) malloc(Data.n * hp.out_dim * sizeof(double));
}

double* BoosterBase::malloc_H(const bool constHessian = true, const double constValue = 0.0) const {
    const size_t n = Data.n * hp.out_dim;
    double* H = (double*) malloc(n * sizeof(double));
    if (constHessian) { std::fill_n(H, n, constValue); }
    return H;
}

void BoosterBase::reset() {
    cache.clear();
    tree.clear();
    trees.clear();
    // TODO: test? should this be Data.n * hp.out_dim???
    //       also, when is this actually used? what's the purpose here?
    //       retrain? then should have a way to set the parameters(!)
    // std::fill_n(Data.Preds, Data.n * hp.out_dim, 0.0);
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

void BoosterBase::dump_nonleaf_nodes(int* trees, double* thresholds) const {
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

void BoosterBase::dump_leaf_nodes(double* leaves) const {
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
