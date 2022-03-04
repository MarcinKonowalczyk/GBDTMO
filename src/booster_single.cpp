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
    int hist_cache = 16
) {
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
    hp.max_caches = hist_cache;

    srand(hp.seed);
    cache = TopkDeque<CacheInfo>(hp.max_caches);
    obj = Objective(hp.loss);
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

    std::vector<Histogram> Hist(hp.inp_dim);
    for (int i = 0; i < hp.inp_dim; i++) { Hist[i] = Histogram(bin_nums[i], 1); }
    hist_all(Train.Orders, Hist);

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
    int out_dim = hp.out_dim;
    G = calloc_G(Train.num * out_dim);
    H = calloc_H(Train.num * out_dim, obj.constHessian, obj.hessian);
    auto early_stoper = EarlyStoper(hp.early_stop == 0 ? num_rounds : hp.early_stop, obj.largerBetter);

    // start training
    for (int i = 0; i < num_rounds; ++i) {
        // Calculate the gradeint and the hessian for all the trees (columns of )
        obj.f_grad(Train, Train.num, out_dim, G, H);
        for (int j = 0; j < out_dim; ++j) {
            growth();
            update();
            if (j < out_dim - 1) {
                G += Train.num;
                H += Train.num;
                Train.Preds += Train.num;
                Eval.Preds += Eval.num;
            } else {
                int pos = (out_dim - 1) * Train.num;
                G -= pos;
                H -= pos;
                Train.Preds -= pos;
                Eval.Preds -= (out_dim - 1) * Eval.num;
            }
        }
        double score = obj.f_score(Train, Train.num, out_dim);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, out_dim);
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
    if (early_stoper.is_continue && Eval.num > 0) {
        showbest(early_stoper.info);
        }
    free(G);
    free(H);
}

void BoosterSingle::predict(const double* features, double* preds, const size_t n, int num_trees = 0) {
    int out_dim = hp.out_dim;
    int max_trees = trees.size() / out_dim;
    num_trees = (num_trees == 0) ? max_trees : std::min(std::max(num_trees, 1), max_trees);
    for (int i = 0; i < num_trees; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            trees[(i*out_dim)+j].pred_value_single(features, preds + (j*n), hp, n);
        }
    }
}