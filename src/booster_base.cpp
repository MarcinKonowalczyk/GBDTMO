#include "booster.h"
#include <limits>

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
    srand(hp.seed);
    cache = TopkDeque<CacheInfo>(hp.max_caches);
    obj = Objective(hp.loss);
    Train.Maps = nullptr;
    Eval.Maps = nullptr;
}

BoosterBase::~BoosterBase() {
    // std::cout << "Hello from the destructor of BoosterBase!\n";
    if (Train.Maps != nullptr) {
        free(Train.Maps);
    }
}

// Set gradient and Hessian matrices
// void BoosterBase::set_gh(double* G, double* H) {
//     this->G = G;
//     this->H = H;
// }

void BoosterBase::set_train_data(double* features, double* preds, int n) {
    Train.Features = features;
    Train.Preds = preds;
    Train.num = n;
    Train.LeafIndex.resize(n);
    // C owns maps but python owns the rest. Train/Maps gets freed in the object destructor.
    Train.Maps = (uint16_t*) malloc(hp.inp_dim * n * sizeof(uint16_t));
    Train.Orders.resize(n);
    for (size_t i = 0; i < n; ++i) { Train.Orders[i] = i; }
}

void BoosterBase::set_eval_data(double* features, double* preds, int n) {
    Eval.Features = features;
    Eval.Preds = preds;
    Eval.num = n;
    Eval.LeafIndex.resize(n);
    Eval.Maps = (uint16_t*) nullptr;
}

void BoosterBase::calc_train_maps() {
    // Calculate Train.Maps and bins
    std::vector<std::vector<double>> bins;
    calculate_histogram_maps(Train.Features, Train.Maps, bins, Train.num, hp.inp_dim, hp.max_bins);

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

void BoosterBase::set_train_label(double* label) { Train.Label_double = label; }
void BoosterBase::set_train_label(int32_t* label) { Train.Label_int32 = label; }
void BoosterBase::set_eval_label(double* label) { Eval.Label_double = label; }
void BoosterBase::set_eval_label(int32_t* label) { Eval.Label_int32 = label; }

void BoosterBase::rebuild_order(
    const std::vector<size_t>& order,
    std::vector<size_t>& order_l,
    std::vector<size_t>& order_r,
    const uint16_t* maps,
    const uint16_t bin
) {
    int count_l = 0, count_r = 0;
    for (size_t i : order) {
        if (maps[i] <= bin) {
            order_l[count_l++] = i;
        } else {
            order_r[count_r++] = i;
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
    // TODO: test? should this be Train.num * hp.out_dim???
    //       also, when is this actually used? what's the purpose here?
    //       retrain? then should have a way to set the parameters(!)
    std::fill_n(Train.Preds, Train.num, hp.base_score);
    std::fill_n(Eval.Preds, Eval.num, hp.base_score);
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
        nonleaf_sizes[i] = trees[i].nonleaf.size();
    }
}

void BoosterBase::dump_leaf_sizes(uint16_t* leaf_sizes) const {
    for (size_t i = 0; i < trees.size(); ++i) {
        leaf_sizes[i] = trees[i].leaf.size();
    }
}

void BoosterBase::dump_nonleaf_nodes(int* trees, double* thresholds) const {
    int i = 0, j = 0;
    for(auto& tree : this->trees) {
        for (auto it : tree.nonleaf) {
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
        for (auto it : tree.leaf) {
            auto node = it.second;
            for (auto& value : node.values) {
                leaves[k++] = value;
            }
        }
    }
}
