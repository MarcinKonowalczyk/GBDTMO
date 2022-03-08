#include "booster.h"

//======================================================================
//                                                                      
//  ##   ##  ######  ##  ##       ####                                
//  ##   ##    ##    ##  ##      ##                                   
//  ##   ##    ##    ##  ##       ###                                 
//  ##   ##    ##    ##  ##         ##                                
//   #####     ##    ##  ######  ####                                 
//                                                                      
//======================================================================

BoosterBase::BoosterBase(HyperParameters p) : hp(p) {
    srand(hp.seed);
    cache = TopkDeque<CacheInfo>(hp.max_caches);
    obj = Objective(hp.loss);
}

// Set gradient and Hessian matrices
void BoosterBase::set_gh(double* G, double* H) {
    this->G = G;
    this->H = H;
}

void BoosterBase::set_train_data(double* features, double* preds, int n) {
    Train.Features = features;
    Train.Preds = preds;
    Train.num = n;
    Train.LeafIndex.resize(n);
    // TODO: C owns maps but python owns the rest
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
    constexpr static auto DOUBLE_MAX = std::numeric_limits<double>::max();
    for (auto& bin : bins) {
        bin_nums.push_back(bin.size() + 1);
        auto tmp = bin;
        tmp.push_back(DOUBLE_MAX);
        bin_values.push_back(tmp);
    }
}

// void BoosterBase::calc_eval_maps() {
//     calculate_histogram_maps(Train.Features, Train.Maps, Train.num, hp.inp_dim, hp.max_bins);
// }

void BoosterBase::set_train_label(double* label) { Train.Label_double = label; }
void BoosterBase::set_eval_label(double* label) { Eval.Label_double = label; }
void BoosterBase::set_train_label(int32_t* label) { Train.Label_int32 = label; }
void BoosterBase::set_eval_label(int32_t* label) { Eval.Label_int32 = label; }

void BoosterBase::rebuild_order(
    std::vector<size_t>& order,
    std::vector<size_t>& order_l,
    std::vector<size_t>& order_r,
    uint16_t* maps,
    uint16_t bin
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

double* BoosterBase::malloc_G(int elements) {
    return (double*) malloc(elements * sizeof(double));
}

double* BoosterBase::malloc_H(int elements, bool constHessian = true, double constValue = 0.0) {
    double* H = (double*) malloc(elements * sizeof(double));
    if (constHessian) {
        for (size_t i = 0; i < elements; ++i) {
            H[i] = constValue;
        }
    }
    return H;
}

// Clear the trees and predictions of the booster
void BoosterBase::reset() {
    // cache.clear(); tree.clear(); // No need to clear these
    trees.clear();
    // if (Train.num > 0) { std::fill_n(Train.Preds, Train.num, hp.base_score); }
    // if (Eval.num > 0) { std::fill_n(Eval.Preds, Eval.num, hp.base_score); }
    std::fill_n(Train.Preds, Train.num, hp.base_score);
    std::fill_n(Eval.Preds, Eval.num, hp.base_score);
}