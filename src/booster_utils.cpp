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

void BoosterUtils::set_bin(uint16_t* bins, double* values) {
    bin_nums.clear();
    bin_values.clear();
    int count = 0;
    for (int i = 0; i < hp.inp_dim; i++) {
        std::vector<double> tmp;
        bin_nums.push_back(bins[i] + 1);
        for (int j = 0; j < bins[i]; j++) {
            tmp.push_back(values[count + j]);
        }
        tmp.push_back(std::numeric_limits<double>::max());
        count += bins[i];
        bin_values.push_back(tmp);
    }
}

// Set gradient and Hessian matrices
void BoosterUtils::set_gh(double* G, double* H) {
    this->G = G;
    this->H = H;
}

void BoosterUtils::set_train_data(uint16_t* maps, double* features, double* preds, int n) {
    Train.Maps = maps;
    Train.Features = features;
    Train.Preds = preds;
    Train.num = n;
    Train.LeafIndex.resize(n);
    Train.Orders.resize(n);
    for (int32_t i = 0; i < n; ++i) { Train.Orders[i] = i; }
}

void BoosterUtils::set_eval_data(uint16_t* maps, double* features, double* preds, int n) {
    Eval.Maps = maps;
    Eval.Features = features;
    Eval.Preds = preds;
    Eval.num = n;
    Eval.LeafIndex.resize(n);
}

void BoosterUtils::set_label(double* x, bool is_train) {
    if (is_train) { Train.Label_double = x; }
    else { Eval.Label_double = x; }
}

void BoosterUtils::set_label(int32_t* x, bool is_train) {
    if (is_train) { Train.Label_int32 = x; }
    else { Eval.Label_int32 = x; }
}

void BoosterUtils::rebuild_order(std::vector<int32_t>& order, std::vector<int32_t>& order_l, std::vector<int32_t>& order_r, uint16_t* maps, uint16_t bin) {
    int count_l = 0, count_r = 0;
    for (auto i : order) {
        if (maps[i] <= bin) {
            order_l[count_l++] = i;
        } else {
            order_r[count_r++] = i;
        }
    }
}

double* BoosterUtils::calloc_G(int elements) {
    return (double*) calloc(elements, sizeof(double));
}

double* BoosterUtils::calloc_H(int elements, bool constHessian = true, double constValue = 0.0) {
    double* H = (double*) calloc(elements, sizeof(double));
    if (constHessian) {
        for (int i = 0; i < elements; ++i) {
            H[i] = constValue;
        }
    }
    return H;
}

// Clear the trees and predictions of the booster
void BoosterUtils::reset() {
    // cache.clear(); tree.clear(); // No need to clear these
    trees.clear();
    // if (Train.num > 0) { std::fill_n(Train.Preds, Train.num, hp.base_score); }
    // if (Eval.num > 0) { std::fill_n(Eval.Preds, Eval.num, hp.base_score); }
    std::fill_n(Train.Preds, Train.num, hp.base_score);
    std::fill_n(Eval.Preds, Eval.num, hp.base_score);
}