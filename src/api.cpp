#include "booster.h"

extern "C" {

void SetBin(BoosterUtils* foo, uint16_t* bins, double* values) {
    foo->set_bin(bins, values);
}
void SetGH(BoosterUtils* foo, double* x, double* y) {
    foo->set_gh(x, y);
}
void Boost(BoosterUtils* foo) {
    foo->growth(); foo->update();
}
void Train(BoosterUtils* foo, int num_rounds) {
    foo->train(num_rounds);
}
void Dump(BoosterUtils* foo, const char* path) {
    foo->dump(path);
}
void Load(BoosterUtils* foo, const char* path) {
    foo->load(path);
}
void Reset(BoosterUtils* foo) {
    foo->reset();
}

//////

void SetTrainData(BoosterUtils* foo, uint16_t* maps, double* features, double* preds, int n) {
    foo->set_train_data(maps, features, preds, n);
}
void SetEvalData(BoosterUtils* foo, uint16_t* maps, double* features, double* preds, int n) {
    foo->set_eval_data(maps, features, preds, n);
}

void SetLabelDouble(BoosterUtils* foo, double* x, bool is_train) {
    foo->set_label(x, is_train);
}
void SetLabelInt(BoosterUtils* foo, int32_t* x, bool is_train) {
    foo->set_label(x, is_train);
}
void Predict(BoosterUtils* foo, double* features, double* preds, int n, int num_trees) {
    foo->predict(features, preds, n, num_trees);
}

//////

BoosterMulti* MultiNew(
        int inp_dim,
        int out_dim,
        const char* name = "mse",
        int max_depth = 5,
        int max_leaves = 32,
        int seed = 0,
        int min_samples = 5,
        double lr = 0.2,
        double reg_l1 = 0.0,
        double reg_l2 = 1.0,
        double gamma = 1e-3,
        double base_score = 0.0f,
        int early_stop = 0,
        bool verbose = true,
        int hist_cache = 16,
        int topk = 0,
        bool one_side = true) {
    return new BoosterMulti(inp_dim, out_dim, name, max_depth, max_leaves,
                            seed, min_samples, lr, reg_l1, reg_l2, gamma, base_score,
                            early_stop, verbose, hist_cache, topk, one_side);
}

BoosterSingle* SingleNew(
        int inp_dim,
        int out_dim,
        const char* name = "mse",
        int max_depth = 5,
        int max_leaves = 32,
        int seed = 0,
        int min_samples = 5,
        double lr = 0.2,
        double reg_l1 = 0.0,
        double reg_l2 = 1.0,
        double gamma = 1e-3,
        double base_score = 0.0f,
        int early_stop = 0,
        bool verbose = true,
        int hist_cache = 16) {
    return new BoosterSingle(inp_dim, out_dim, name, max_depth, max_leaves,
                             seed, min_samples, lr, reg_l1,
                             reg_l2, gamma, base_score,
                             early_stop, verbose, hist_cache);
}

}