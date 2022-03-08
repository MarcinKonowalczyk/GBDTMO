#include "booster.h"
#include "histogram.h"

//================================================================================
//                                                                                
//   ####  ##  ##     ##   ####    ##      #####                                
//  ##     ##  ####   ##  ##       ##      ##                                   
//   ###   ##  ##  ## ##  ##  ###  ##      #####                                
//     ##  ##  ##    ###  ##   ##  ##      ##                                   
//  ####   ##  ##     ##   ####    ######  #####                                
//                                                                                
//================================================================================

BoosterSingle::BoosterSingle(HyperParameters hp) : BoosterBase(hp) {};

void BoosterSingle::get_score_opt(Histogram& Hist, double& opt, double& score_sum) {
    double gr = Hist.g[Hist.g.size() - 1];
    double hr = Hist.h[Hist.h.size() - 1];
    opt = CalWeight(gr, hr, hp.reg_l1, hp.reg_l2);
    score_sum = CalScore(gr, hr, hp.reg_l1, hp.reg_l2);
}

// void histogram_single(
//     std::vector<size_t>& order,
//     Histogram& Hist,
//     uint16_t* maps,
//     double* G,
//     double* H
// )

void BoosterSingle::hist_all(
    std::vector<size_t>& order,
    std::vector<Histogram>& Hist
) {
    for (size_t i = 0; i < hp.inp_dim; ++i) {
        histogram_single(order, Hist[i], Train.Maps + i * Train.num, G, H);
    }
}

void BoosterSingle::boost_column(Histogram& Hist, int column) {
    int max_bins = Hist.count.size() - 1;
    double gr, hr;
    double gain = 0.0f, tmp;
    int row_bins = -1;
    for (size_t i = 0; i < max_bins; ++i) {
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
    for (size_t i = 0; i < hp.inp_dim; ++i) {
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

    std::vector<size_t> order_l(rows_l), order_r(rows_r);
    rebuild_order(info->order, order_l, order_r, Train.Maps + info->split.column * Train.num, info->split.bin);
    std::vector<Histogram> Hist_l(hp.inp_dim), Hist_r(hp.inp_dim);

    if (rows_l >= rows_r) {
        for (size_t i = 0; i < hp.inp_dim; ++i) { Hist_r[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_r, Hist_r);
        for (size_t i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_r[i]; }
        Hist_l.assign(info->hist.begin(), info->hist.end());
    } else {
        for (size_t i = 0; i < hp.inp_dim; ++i) { Hist_l[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_l, Hist_l);
        for (size_t i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_l[i]; }
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
    for (size_t i = 0; i < hp.inp_dim; ++i) { Hist[i] = Histogram(bin_nums[i], 1); }
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
    tree.pred_value_single(Train.Features, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_single(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(tree);
}

void BoosterSingle::train(int num_rounds) {

    int out_dim = hp.out_dim;
    G = malloc_G(Train.num * out_dim);
    H = malloc_H(Train.num * out_dim, obj.constHessian, obj.hessian);
    auto early_stoper = EarlyStoper(hp.early_stop == 0 ? num_rounds : hp.early_stop, obj.largerBetter);

    // start training
    for (size_t i = 0; i < num_rounds; ++i) {
        // Calculate the gradeint and the hessian for all the trees (columns of )
        obj.f_grad(Train, Train.num, out_dim, G, H);
        for (size_t j = 0; j < out_dim; ++j) {
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
    if (early_stoper.is_continue && Eval.num > 0) { showbest(early_stoper.info); }
    free(G);
    free(H);
}

void BoosterSingle::predict(const double* features, double* preds, const size_t n, int num_trees = 0) {
    int max_trees = trees.size() / hp.out_dim;
    num_trees = (num_trees == 0) ? max_trees : std::min(std::max(num_trees, 1), max_trees);
    // for (size_t i = 0; i < num_trees; ++i) {
    //     for (int j = 0; j < hp.out_dim; ++j) {
    //         trees[(i*hp.out_dim)+j].pred_value_single(features, preds + (j*n), hp, n);
    //     }
    // }
    for (size_t i = 0; i < num_trees; ++i) {
        for (size_t j = 0; j < hp.out_dim; ++j) {
            trees[(i*hp.out_dim)+j].pred_value_single(features, preds + (j*n), hp, n);
        }
    }
}