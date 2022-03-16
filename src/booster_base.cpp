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
    // Train.Maps = nullptr;
    // Eval.Maps = nullptr;
}

// BoosterBase::~BoosterBase() {
//     if (Train.Maps != nullptr) {
//         free(Train.Maps);
//     }
// }

//=======================================================================================
//                                                                                       
//   ####  #####  ######  ######  #####  #####     ####                                
//  ##     ##       ##      ##    ##     ##  ##   ##                                   
//   ###   #####    ##      ##    #####  #####     ###                                 
//     ##  ##       ##      ##    ##     ##  ##      ##                                
//  ####   #####    ##      ##    #####  ##   ##  ####                                 
//                                                                                       
//=======================================================================================

void BoosterBase::set_train_data(double* features, double* preds, size_t n) {
    Train.Features = features;
    Train.Preds = preds;
    Train.n = n;
}

// void BoosterBase::set_eval_data(double* features, double* preds, int n) {
//     Eval.Features = features;
//     Eval.Preds = preds;
//     Eval.num = n;
//     Eval.Maps = (uint16_t*) nullptr;
// }

void BoosterBase::set_train_label(double* label) { Train.Label_double = label; }
void BoosterBase::set_train_label(int32_t* label) { Train.Label_int32 = label; }
// void BoosterBase::set_eval_label(double* label) { Eval.Label_double = label; }
// void BoosterBase::set_eval_label(int32_t* label) { Eval.Label_int32 = label; }

//===========================================================================================================
//                                                                                                           
//   ####    ###    ##       ####        ###    ###    ###    #####    ####                                
//  ##      ## ##   ##      ##           ## #  # ##   ## ##   ##  ##  ##                                   
//  ##     ##   ##  ##      ##           ##  ##  ##  ##   ##  #####    ###                                 
//  ##     #######  ##      ##           ##      ##  #######  ##         ##                                
//   ####  ##   ##  ######   ####        ##      ##  ##   ##  ##      ####                                 
//                                                                                                           
//===========================================================================================================

void BoosterBase::calc_train_maps() {
    // Calculate Train.Maps and bins
    std::vector<std::vector<double>> bins;

    Train.train_maps.resize(hp.inp_dim);
    for (size_t i = 0; i < hp.inp_dim; ++i) {
        // Get features column
        std::vector<double> features_column;
        features_column.reserve(Train.n);
        for (size_t j = 0; j < Train.n; ++j) { features_column.push_back(Train.Features[i + j*hp.inp_dim]); }

        // Construct bins for the column
        std::vector<double> bins_column;
        construct_bin_column(features_column, bins_column, hp.max_bins);
        bins.push_back(bins_column);

        // Map features in the particular column
        // NOTE: This means that maps are column-major
        Train.train_maps[i].resize(Train.n);
        map_bin_column(features_column, Train.train_maps[i], bins_column);
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

void BoosterBase::calc_eval_indices() {

    Train.n_eval = 0;
    Train.n_train = Train.n;

    Train.train_order.resize(Train.n_train);
    std::iota(Train.train_order.begin(), Train.train_order.end(), 0);

    if (hp.eval_fraction >= 0.0) {
        Train.n_eval = static_cast<size_t>(hp.eval_fraction * Train.n);
        Train.n_train -= Train.n_eval;
        std::cout << "calc_eval_indices. Train.n_eval = " << Train.n_eval << "\n";
        std::cout << "calc_eval_indices. Train.n_train = " << Train.n_train << "\n";
        
        // TODO: this can be done so much better!!
        std::vector<size_t> perm(Train.n);
        std::iota(perm.begin(), perm.end(), 0);

        std::random_shuffle(perm.begin(), perm.end());

        // Trim order to the train_subset
        size_t n_train = Train.n_train;
        Train.train_order.erase(std::remove_if(
            Train.train_order.begin(), Train.train_order.end(),
            [&perm, n_train](size_t o) { return perm[o] >= n_train; }
        ), Train.train_order.end());
    }
}

void BoosterBase::rebuild_order(
    const std::vector<size_t>& order,
    std::vector<size_t>& order_l,
    std::vector<size_t>& order_r,
    const size_t split_column,
    const uint16_t bin
) const {
    const auto map_column = Train.train_maps[split_column];
    int count_l = 0, count_r = 0;
    for (size_t o : order) {
        if (map_column[o] <= bin) {
            if (count_l >= order_l.size()) {
                std::cout << "count_l too large! order_l.size() = " << order_l.size() << " but count_l = " << count_l << "!\n";
                std::cout << " order.size() = " << order.size() << "\n";
                std::cout << " o = " << o << "\n";
                std::cout << " map_column[o] = " << map_column[o] << "\n";
                std::cout << " split_column = " << split_column << "\n";
                std::cout << " bin = " << bin << "\n";
                }
            order_l[count_l++] = o;
        } else {
            if (count_r >= order_r.size()) { std::cout << "count_r too large! order_r.size() = " << order_r.size() << " but count_r = " << count_r << "!\n"; }
            order_r[count_r++] = o;
        }
    }
}

double* BoosterBase::malloc_G(size_t elements) {
    return (double*) malloc(elements * sizeof(double));
}

double* BoosterBase::malloc_H(size_t elements, bool constHessian = true, double constValue = 0.0) {
    double* H = (double*) malloc(elements * sizeof(double));
    if (constHessian) {
        std::fill_n(H, elements, constValue);
        // for (size_t i = 0; i < elements; ++i) {
        //     H[i] = constValue;
        // }
    }
    return H;
}

void BoosterBase::reset() {
    cache.clear();
    tree.clear();
    trees.clear();
    // TODO: test? should this be Train.n * hp.out_dim???
    //       also, when is this actually used? what's the purpose here?
    //       retrain? then should have a way to set the parameters(!)
    // std::fill_n(Train.Preds, Train.n * hp.out_dim, 0.0);
    // std::fill_n(Eval.Preds, Eval.num * hp.out_dim, 0.0);
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
