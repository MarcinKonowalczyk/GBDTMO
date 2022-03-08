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

void BoosterBase::set_bin(uint16_t* n_bins, double* values) {
    bin_nums.clear();
    bin_values.clear();
    size_t count = 0;
    for (size_t i = 0; i < hp.inp_dim; ++i) {
        std::vector<double> tmp;
        bin_nums.push_back(n_bins[i] + 1);
        for (size_t j = 0; j < n_bins[i]; ++j) { tmp.push_back(values[count + j]); }
        tmp.push_back(std::numeric_limits<double>::max());
        count += n_bins[i];
        bin_values.push_back(tmp);
    }
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

// #include <iomanip>
// TODO: What frees this memory...?
void BoosterBase::calc_train_maps() {
    // uint16_t* maps = (uint16_t*) malloc(Train.num * hp.inp_dim * sizeof(double));


    // uint16_t* new_maps = (uint16_t*) calloc(Train.num * hp.inp_dim, sizeof(double));

    std::vector<std::vector<double>> bins;
    calculate_histogram_maps(Train.Features, Train.Maps, bins, Train.num, hp.inp_dim, hp.max_bins);
    // Train.Maps = new_maps;

    // std::cout << "Train.maps[:50] = [ ";
    // std::cout << std::fixed << std::setprecision(2);
    // for(int i = 0; i < 50; ++i) { std::cout << Train.Maps[i] << " "; }
    // std::cout << "]\n";

    // std::cout << "new_maps  [:50] = [ ";
    // std::cout << std::fixed << std::setprecision(2);
    // for(int i = 0; i < 50; ++i) { std::cout << new_maps[i] << " "; }
    // std::cout << "]\n";

    std::vector<uint16_t> n_bins;
    std::vector<double> values;
    // n_bins.resize(bins.size());
    for(auto& bin : bins) {
        n_bins.push_back(bin.size());
        for(auto& value : bin) {
            values.push_back(value);
        }
    }
    std::cout << "got here" << std::endl;
    std::cout << "bin_nums.size() = " << n_bins.size() << std::endl;
    std::cout << "bin_values.size() = " << values.size() << std::endl; 

    set_bin(&n_bins[0], &values[0]);
    std::cout << "and also got here" << std::endl;


    // set_bin(uint16_t* n_bins, double* values) {
    //     bin_nums.clear();
    //     bin_values.clear();
    //     size_t count = 0;
    //     for (size_t i = 0; i < hp.inp_dim; ++i) {
    //         std::vector<double> tmp;
    //         bin_nums.push_back(n_bins[i] + 1);
    //         for (size_t j = 0; j < n_bins[i]; ++j) { tmp.push_back(values[count + j]); }
    //         tmp.push_back(std::numeric_limits<double>::max());
    //         count += n_bins[i];
    //         bin_values.push_back(tmp);
    //     }
    // }

    // std::vector<uint16_t> bin_nums;
    // std::vector<std::vector<double>> bin_values;

    // std::cout << "maps[50:] = [ ";
    // for(size_t i = 0; i < 50; ++i) { std::cout << maps[i] << " "; }
    // std::cout << "]\n";

    // std::cout << "maps[:50] = [ ";
    // for(size_t i = 0; i < 50; ++i) { std::cout << maps[hp.inp_dim*Train.num-i] << " "; }
    // std::cout << "]\n";

    // Train.Maps = maps;
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