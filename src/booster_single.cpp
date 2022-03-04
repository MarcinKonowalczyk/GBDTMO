#include "booster.h"

//================================================================================
//                                                                                
//   ####  ##  ##     ##   ####    ##      #####                                
//  ##     ##  ####   ##  ##       ##      ##                                   
//   ###   ##  ##  ## ##  ##  ###  ##      #####                                
//     ##  ##  ##    ###  ##   ##  ##      ##                                   
//  ####   ##  ##     ##   ####    ######  #####                                
//                                                                                
//================================================================================

BoosterSingle::BoosterSingle(
    int inp_dim,
    int out_dim,
    const char* loss = "mse",
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
    int hist_cache = 16)
{
    hp.inp_dim = inp_dim;
    hp.out_dim = out_dim;
    hp.loss = loss;
    hp.max_depth = max_depth;
    hp.max_leaves = max_leaves;
    hp.seed = seed;
    hp.min_samples = min_samples;
    hp.lr = lr;
    hp.reg_l1 = reg_l1;
    hp.reg_l2 = reg_l2;
    hp.gamma = gamma;
    hp.base_score = base_score;
    hp.early_stop = early_stop;
    hp.verbose = verbose;
    hp.Max_caches = hist_cache;

    srand(hp.seed);
    cache = TopkDeque<CacheInfo>(hp.Max_caches);
    obj = Objective(hp.loss);
}

void BoosterSingle::reset() {
    trees.clear();
    if (Train.num > 0) { std::fill_n(Train.Preds, Train.num, hp.base_score); }
    if (Eval.num > 0) { std::fill_n(Eval.Preds, Eval.num, hp.base_score); }

}

void BoosterSingle::get_score_opt(Histogram& Hist, double& opt, double& score_sum) {
    double gr = Hist.g[Hist.g.size() - 1];
    double hr = Hist.h[Hist.h.size() - 1];
    opt = CalWeight(gr, hr, hp.reg_l1, hp.reg_l2);
    score_sum = CalScore(gr, hr, hp.reg_l1, hp.reg_l2);
}

void BoosterSingle::hist_all(std::vector<int32_t>& order, std::vector<Histogram>& Hist) {
    for (int i = 0; i < hp.inp_dim; i++) {
        histogram_single(order, Hist[i], Train.Maps + i * Train.num, G, H);
    }
}

void BoosterSingle::boost_column(Histogram& Hist, int column) {
    int max_bins = Hist.count.size() - 1;
    double gr, hr;
    double gain = 0.0f, tmp;
    int row_bins = -1;
    for (int i = 0; i < max_bins; i++) {
        gr = Hist.g[max_bins] - Hist.g[i];
        hr = Hist.h[max_bins] - Hist.h[i];
        tmp = CalScore(Hist.g[i], Hist.h[i], hp.reg_l1, hp.reg_l2) + CalScore(gr, hr, hp.reg_l1, hp.reg_l2);
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5f;
    if (gain > meta.gain) {
        meta.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
}

void BoosterSingle::boost_all(std::vector<Histogram>& Hist) {
    meta.reset();
    for (int i = 0; i < hp.inp_dim; ++i) {
        boost_column(Hist[i], i);
    }
}

void BoosterSingle::build_tree_best() {
    if (tree.leaf_num >= hp.max_leaves) { return; }

    auto info = &cache.data[0];
    int parent = info->node;
    int depth = info->depth;

    int rows_l = info->hist[info->split.column].count[info->split.bin];
    int rows_r = info->order.size() - rows_l;

    std::vector<int32_t> order_l(rows_l), order_r(rows_r);
    rebuild_order(info->order, order_l, order_r,
                  Train.Maps + Train.num * info->split.column, info->split.bin);

    std::vector<Histogram> Hist_l(hp.inp_dim), Hist_r(hp.inp_dim);

    if (rows_l >= rows_r) {
        for (int i = 0; i < hp.inp_dim; i++) { Hist_r[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_r, Hist_r);
        for (int i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_r[i]; }
        Hist_l.assign(info->hist.begin(), info->hist.end());
    } else {
        for (int i = 0; i < hp.inp_dim; i++) { Hist_l[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_l, Hist_l);
        for (int i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_l[i]; }
        Hist_r.assign(info->hist.begin(), info->hist.end());
    }
    cache.pop_front();

    if (rows_l >= hp.min_samples) {
        get_score_opt(Hist_l[rand() % hp.inp_dim], Opt, Score_sum);
        boost_all(Hist_l);

        if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
            auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
            tree.add_nonleaf(node, true);
            cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_l, Hist_l));
        } else {
            auto node = LeafNode(1);
            node.Update(parent, Opt);
            tree.add_leaf(node, true);
        }
    }
    order_l.clear();
    Hist_l.clear();

    if (tree.leaf_num >= hp.max_leaves) { return; }
    if (rows_r >= hp.min_samples) {
        get_score_opt(Hist_r[rand() % hp.inp_dim], Opt, Score_sum);
        boost_all(Hist_r);

        if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
            auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
            tree.add_nonleaf(node, false);
            cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_r, Hist_r));
        } else {
            auto node = LeafNode(hp.out_dim);
            node.Update(parent, Opt);
            tree.add_leaf(node, false);
        }
    }
    order_r.clear();
    Hist_r.clear();

    if (!cache.empty()) { build_tree_best(); }
}

void BoosterSingle::growth() {
    tree.clear();
    cache.clear();

    // std::cout << "hp.inp_dim = " << hp.inp_dim << std::endl;
    // std::cout << "bin_nums = [";
    // for (int i = 0; i < hp.inp_dim; i++) { std::cout << bin_nums[i] << ", "; }
    // std::cout << "]" << std::endl;
    // std::cout << "Train.Orders.size() = " << Train.Orders.size() << std::endl;
    // std::cout << "Train.Orders[0] = " << Train.Orders[0] << std::endl;
    // std::cout << "Train.Orders[1] = " << Train.Orders[1] << std::endl;
    // std::cout << "Train.Orders[2] = " << Train.Orders[2] << std::endl;
    // std::cout << "Train.Orders[3] = " << Train.Orders[3] << std::endl;
    // std::cout << "Train.Orders[-1] = " << Train.Orders[Train.Orders.size()-1] << std::endl;
    
    std::vector<Histogram> Hist(hp.inp_dim);
    for (int i = 0; i < hp.inp_dim; i++) { Hist[i] = Histogram(bin_nums[i], 1); }
    hist_all(Train.Orders, Hist);

    // std::cout << "Hist[0].count.size() = " << Hist[0].count.size() << std::endl;
    // std::cout << "Hist[0].count[16] = " << Hist[0].count[16] << std::endl;

    get_score_opt(Hist[rand() % hp.inp_dim], Opt, Score_sum);
    boost_all(Hist);

    if (meta.column > -1 & meta.gain > -10.0f) {
        auto node = NonLeafNode(-1, meta.column, meta.bin, meta.threshold);
        tree.add_nonleaf(node, true);
        cache.push(CacheInfo(-1, 0, meta, Train.Orders, Hist));
        build_tree_best();
    } else {
        auto node = LeafNode(1);
        node.Update(-1, Opt);
        tree.add_leaf(node, true);
    }
}

void BoosterSingle::update() {
    tree.shrinkage(hp.lr);
    // tree.pred_value_single(Train.Maps, Train.Preds, hp, Train.num);
    tree.pred_value_single(Train.Features, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_single(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(tree);
}

void BoosterSingle::train(int num_rounds) {
    // initialize gradient and hessian
    G = calloc_G(Train.num);
    H = calloc_H(Train.num, obj.constHessian, obj.hessian);

    int round = hp.early_stop;
    if (round == 0) { round = num_rounds; }
    auto early_stoper = EarlyStoper(round, obj.largerBetter);

    // start training
    for (int i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, Train.num, 1, G, H);
        growth();
        update();
        double score = obj.f_score(Train, Train.num, 1);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, 1);
            if (hp.verbose) { showloss(score, metric, i); }
            early_stoper.push(std::make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.info;
                int round = std::get<1>(info);
                showbest(std::get<0>(info), round);
                trees.resize(round + 1);;
                break;
            }
        } else {
            if (hp.verbose) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && Eval.num > 0) { showbest(early_stoper.info); }
    free(G);
    free(H);
}

// void BoosterSingle::train_multi(int num_rounds) {
//     std::cout << "hello from hacked version of train_multi" << std::endl;
//     // hp.out_dim = 1;
//     int out_dim = hp.out_dim;
//     for (int j = 0; j < out_dim; ++j) {
//         // if (j < out_dim - 1) {
//         //     Train.Preds += Train.num;
//         //     Train.Label_double += Train.num;
//         //     Train.Label_int32 += Train.num;
//         //     if (Eval.num > 0) {
//         //         Eval.Preds += Eval.num;
//         //         Eval.Label_double += Eval.num;
//         //         Eval.Label_int32 += Eval.num;
//         //     }
//         // } else {
//         //     int pos = (out_dim - 1) * Train.num;
//         //     Train.Preds -= pos;
//         //     Train.Label_double -= pos;
//         //     Train.Label_int32 -= pos;
//         //     if (Eval.num > 0) {
//         //         int pos = (out_dim - 1) * Eval.num;
//         //         Eval.Preds -= pos;
//         //         Eval.Label_double -= pos;
//         //         Eval.Label_int32 -= pos;
//         //     }
//         // }

//         // initialize gradient and hessian
//         G = calloc_G(Train.num);
//         H = calloc_H(Train.num, obj.constHessian, obj.hessian);

//         int round = hp.early_stop;
//         if (round == 0) { round = num_rounds; }
//         auto early_stoper = EarlyStoper(round, obj.largerBetter);

//         // start training
//         for (int i = 0; i < num_rounds; ++i) {
//             obj.f_grad(Train, Train.num, 1, G, H);
//             growth();
//             update();
//             double score = obj.f_score(Train, Train.num, 1);
//             // if (Eval.num > 0) {
//             //     double metric = obj.f_metric(Eval, Eval.num, 1);
//             //     if (hp.verbose) { showloss(score, metric, i); }
//             //     early_stoper.push(std::make_pair(metric, i));
//             //     if (!early_stoper.is_continue) {
//             //         auto info = early_stoper.info;
//             //         int round = std::get<1>(info);
//             //         showbest(std::get<0>(info), round);
//             //         trees.resize(round + 1);;
//             //         break;
//             //     }
//             // } else {
//             //     if (hp.verbose) { showloss(score, i); }
//             // }
//             showloss(score, i);
//         }
//         if (early_stoper.is_continue && Eval.num > 0) { showbest(early_stoper.info); }
//         free(G);
//         free(H);
//     }
//     std::cout << "trees.size() = " << trees.size() << std::endl;
// }

// void BoosterSingle::train_multi(int num_rounds) {
//     int out_dim = hp.out_dim;
//     std::cout << "hp.inp_dim = " << hp.inp_dim << std::endl;
//     std::cout << "Train.num = " << Train.num << std::endl;
//     std::cout << "num_rounds = " << num_rounds << std::endl;
//     std::cout << "out_dim = " << out_dim << std::endl;
//     // G = calloc_G(Train.num * out_dim);
//     // H = calloc_H(Train.num * out_dim, obj.constHessian, obj.hessian);
//     G = calloc_G(Train.num);
//     H = calloc_H(Train.num, obj.constHessian, obj.hessian);
    
//     std::cout << "hp.early_stop = " << hp.early_stop << std::endl;
//     std::cout << "obj.largerBetter = " << obj.largerBetter << std::endl;
//     auto early_stoper = EarlyStoper(hp.early_stop, obj.largerBetter);
//     if (hp.early_stop == 0) { early_stoper = EarlyStoper(num_rounds, obj.largerBetter); }

//     // start training
//     for (int i = 0; i < num_rounds; ++i) { // 11
//         // std::cout << "G[0], G[-1] = " << G[0] << ", " << G[Train.num * out_dim - 1] << std::endl;
//         // std::cout << "H[0], H[-1] = " << H[0] << ", " << H[Train.num * out_dim - 1] << std::endl;
//         for (int j = 0; j < out_dim; ++j) { // 2
//             obj.f_grad(Train, Train.num, 1, G, H);
//             std::cout << "G[0,-1] = " << G[0] << ", " << G[Train.num-1] << std::endl;
//             growth();
//             update();
//             if (j < out_dim - 1) {
//                 // G += Train.num;
//                 // H += Train.num;
//                 Train.Preds += Train.num;
//                 Train.Label_double += Train.num;
//                 Train.Label_int32 += Train.num;
//                 if (Eval.num > 0) { Eval.Preds += Eval.num; }
//             } else {
//                 int pos = (out_dim - 1) * Train.num;
//                 // G -= pos;
//                 // H -= pos;
//                 Train.Preds -= pos;
//                 Train.Label_double -= pos;
//                 Train.Label_int32 -= pos;
//                 if (Eval.num > 0) { Eval.Preds -= (out_dim - 1) * Eval.num; }
//             }
//         }

//         double score = obj.f_score(Train, Train.num, out_dim);
//         if (Eval.num > 0) {
//             double metric = obj.f_metric(Eval, Eval.num, out_dim);
//             if (hp.verbose) { showloss(score, metric, i); }
//             early_stoper.push(std::make_pair(metric, i));
//             if (!early_stoper.is_continue) {
//                 auto info = early_stoper.info;
//                 int round = std::get<1>(info);
//                 showbest(std::get<0>(info), round);
//                 trees.resize(round + 1);
//                 break;
//             }
//         } else {
//             if (hp.verbose) { showloss(score, i); }
//         }

//     }
//     if (early_stoper.is_continue && Eval.num > 0) { showbest(early_stoper.info); }
//     free(G);
//     free(H);
// }

void BoosterSingle::train_multi(int num_rounds) {
    int out_dim = hp.out_dim;
    G = calloc_G(Train.num * out_dim);
    H = calloc_H(Train.num * out_dim, obj.constHessian, obj.hessian);
    auto early_stoper = EarlyStoper(hp.early_stop == 0 ? num_rounds : hp.early_stop, obj.largerBetter);

    // start training
    for (int i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, Train.num, out_dim, G, H);
        for (int j = 0; j < out_dim; ++j) {
            growth();
            update();
            if (j < out_dim - 1) {
                G += Train.num;
                H += Train.num;
                Train.Preds += Train.num;
                if (Eval.num > 0) { Eval.Preds += Eval.num; }
            } else {
                int pos = (out_dim - 1) * Train.num;
                G -= pos;
                H -= pos;
                Train.Preds -= pos;
                if (Eval.num > 0) { Eval.Preds -= (out_dim - 1) * Eval.num; }
            }
        }
        double score = obj.f_score(Train, Train.num, out_dim);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, out_dim);
            // std::cout << "metric = " << metric << std::endl;
            if (hp.verbose) { showloss(score, metric, i); }
            early_stoper.push(std::make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.info;
                int round = std::get<1>(info);
                showbest(std::get<0>(info), round);
                trees.resize(round + 1);
                break;
            }
        } else {
            if (hp.verbose) { showloss(score, i); }
        }
        
    }
    if (early_stoper.is_continue && Eval.num > 0) { showbest(early_stoper.info); }
    free(G);
    free(H);
}

void BoosterSingle::predict(const double* features, double* preds, const size_t n, int num_trees = 0) {
    num_trees = num_trees == 0 ? int(trees.size()) : std::min(num_trees, int(trees.size()));
    for (int i = 0; i < num_trees; ++i) {
        trees[i].pred_value_single(features, preds, hp, n);
    }
}


// void BoosterSingle::predict_multi(const double* features, double* preds, const size_t n, const int out_dim, int num_trees = 0) {
//     std::cout << "hello from BoosterSingle::predict_multi" << std::endl;
//     int max_trees = trees.size() / out_dim;
//     num_trees = (num_trees == 0) ? max_trees : std::min(std::max(num_trees, 1), max_trees);
//     for (int i = 0; i < num_trees; ++i) {
//         for (int j = 0; j < out_dim; ++j) {
//             // std::cout << "t = " << t << " i = " << i << " j = " << j << std::endl;
//             trees[(i*out_dim)+j].pred_value_single(features, preds + (j*n), hp, n);
//         }
//     }
//     // for (int j = 0; j < out_dim; ++j) {
//     //     for (int i = 0; i < num_trees; ++i) {
//     //         trees[(j*num_trees)+i].pred_value_single(features, preds + (j*n), hp, n);
//     //     }
//     // }
// }

void BoosterSingle::predict_multi(const double* features, double* preds, const size_t n, const int out_dim, int num_trees = 0) {
    std::cout << "hello from BoosterSingle::predict_multi" << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "out_dim = " << out_dim << std::endl;
    int max_trees = trees.size() / out_dim;
    num_trees = (num_trees == 0) ? max_trees : std::min(std::max(num_trees, 1), max_trees);
    int t = 0;
    for (int i = 0; i < num_trees; ++i) {
        int start = 0;
        for (int j = 0; j < out_dim; ++j) {
            // std::cout << "t = " << t << " i = " << i << " j = " << j << std::endl;
            trees[t+j].pred_value_single(features, preds + start, hp, n);
            start += n;
        }
        t += out_dim;
    }
}

// void BoosterSingle::predict_multi(const double* features, double* preds, const size_t n, const int out_dim, int num_trees = 0) {
//     std::cout << "hello from BoosterSingle::predict_multi" << std::endl;
//     num_trees = num_trees == 0 ? int(trees.size()) : std::max(std::min(num_trees, int(trees.size())), out_dim);
//     std::cout << "trees.size = " << trees.size() << std::endl;
//     std::cout << "num_trees = " << num_trees << std::endl;
//     std::cout << "n = " << n << std::endl;
//     // int t = 0;
//     int num_rounds = num_trees / out_dim;
//     std::cout << "num_rounds = " << num_rounds << std::endl;
//     for (int i = 0; i < num_rounds; ++i) { // 11
//         for (int j = 0; j < out_dim; ++j) { // 2
//             std::cout << "-----" << std::endl;
//             // std::cout << "i = " << i << std::endl;
//             std::cout << "j = " << j << std::endl;
//             std::cout << "j + i*out_dim = " << j + i*out_dim << std::endl;
//             // std::cout << "&trees[i*out_dim+j] = " << &trees[i*out_dim+j] << std::endl;
//             // trees[i*out_dim+j].pred_value_single(features, preds, hp, n);
//             // trees[j + i*out_dim].pred_value_single(features, preds + (j*n), hp, n);
//             trees[j + i*out_dim].pred_value_single(features, preds + (j*n), hp, n);
//         }
//     }
// }