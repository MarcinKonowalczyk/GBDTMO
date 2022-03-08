#include "booster.h"
#include "histogram.h"

//===========================================================================
//                                                                           
//  ###    ###  ##   ##  ##      ######  ##                                
//  ## #  # ##  ##   ##  ##        ##    ##                                
//  ##  ##  ##  ##   ##  ##        ##    ##                                
//  ##      ##  ##   ##  ##        ##    ##                                
//  ##      ##   #####   ######    ##    ##                                
//                                                                           
//===========================================================================

BoosterMulti::BoosterMulti(HyperParameters hp) : BoosterBase(hp) {
    Score.resize(hp.out_dim);
    if (hp.topk > 0) {
        OptPair.resize(hp.topk);
    } else {
        Opt.resize(hp.out_dim);
    }
}

void BoosterMulti::get_score_opt(
    Histogram& Hist,
    std::vector<double>& opt,
    std::vector<double>& score,
    double& score_sum
) {
    double* gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double* hr = &Hist.h[Hist.h.size() - hp.out_dim];
    CalWeight(opt, gr, hr, hp.reg_l1, hp.reg_l2);
    CalScore(score, gr, hr, hp.reg_l1, hp.reg_l2);
    score_sum = 0.0f;
    for (size_t i = 0; i < hp.out_dim; ++i) {
        score_sum += score[i];
    }
}

void BoosterMulti::get_score_opt(
    Histogram& Hist,
    std::vector<std::pair<double, int>>& opt,
    std::vector<double>& score,
    double& score_sum
) {
    double* gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double* hr = &Hist.h[Hist.h.size() - hp.out_dim];
    TopkPriority<std::pair<double, int>> score_k(hp.topk);
    for (size_t i = 0; i < hp.out_dim; ++i) {
        score[i] = CalScore(gr[i], hr[i], hp.reg_l1, hp.reg_l2);
        score_k.push(std::make_pair(score[i], i));
    }
    opt.resize(0);
    score_sum = 0.0f;
    while (!score_k.empty()) {
        auto top = score_k.top();
        score_sum += std::get<0>(top);
        int k = std::get<1>(top);
        opt.push_back(std::make_pair(CalWeight(gr[k], hr[k], hp.reg_l1, hp.reg_l2), k));
        score_k.pop();
    }
}

void BoosterMulti::hist_all(
    std::vector<size_t>& order,
    std::vector<Histogram>& Hist
) {
    for (size_t i = 0; i < hp.inp_dim; ++i) {
        histogram_multi(order, Hist[i], Train.Maps + i*Train.num, G, H, hp.out_dim);
    }
}

void BoosterMulti::boost_column_full(Histogram& Hist, int column) {
    double* gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double* hr = &Hist.h[Hist.h.size() - hp.out_dim];
    double gain = 0.0f, tmp;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.count.size() - 1;

    for (size_t i = 0; i < max_bins; ++i) {
        tmp = 0.0f;
        for (size_t j = 0; j < hp.out_dim; ++j) {
            ind_l_tmp = ind_l + j;
            tmp += CalScore(Hist.g[ind_l_tmp], Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2) + \
                   CalScore(gr[j] - Hist.g[ind_l_tmp], hr[j] - Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
        }
        ind_l += hp.out_dim;
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5 / hp.out_dim;
    if (gain > meta.gain) {
        meta.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
}

void BoosterMulti::boost_column_topk_two_side(Histogram& Hist, int column) {
    double* gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double* hr = &Hist.h[Hist.h.size() - hp.out_dim];

    double gain = 0.0f, tmp;
    double score_l, score_r;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.count.size() - 1;

    auto pq_l = TopkPriority<std::pair<double, int>>(hp.topk);
    auto pq_r = TopkPriority<std::pair<double, int>>(hp.topk);

    for (size_t i = 0; i < max_bins; ++i) {
        for (size_t j = 0; j < hp.out_dim; ++j) {
            ind_l_tmp = ind_l + j;
            score_l = CalScore(Hist.g[ind_l_tmp], Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            score_r = CalScore(gr[j] - Hist.g[ind_l_tmp], hr[j] - Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            pq_l.push(std::make_pair(score_l, j));
            pq_r.push(std::make_pair(score_r, j));
        }
        ind_l += hp.out_dim;
        tmp = 0.0f;
        while (!pq_l.empty()) {
            tmp += std::get<0>(pq_l.top());
            pq_l.pop();
        }
        while (!pq_r.empty()) {
            tmp += std::get<0>(pq_r.top());
            pq_r.pop();
        }
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5 / hp.topk;
    if (gain > meta.gain) {
        meta.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
}

void BoosterMulti::boost_column_topk_one_side(Histogram& Hist, int column) {
    double* gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double* hr = &Hist.h[Hist.h.size() - hp.out_dim];

    double gain = 0.0f, tmp;
    double score_l, score_r;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.count.size() - 1;

    auto pq = TopkPriority<std::pair<double, int>>(hp.topk);

    for (size_t i = 0; i < max_bins; ++i) {
        for (size_t j = 0; j < hp.out_dim; ++j) {
            ind_l_tmp = ind_l + j;
            score_l = CalScore(Hist.g[ind_l_tmp], Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            score_r = CalScore(gr[j] - Hist.g[ind_l_tmp], hr[j] - Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            pq.push(std::make_pair(score_l + score_r, j));
        }
        ind_l += hp.out_dim;
        tmp = 0.0f;
        while (!pq.data.empty()) {
            tmp += std::get<0>(pq.data.top());
            pq.data.pop();
        }
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5 / hp.topk;
    if (gain > meta.gain) {
        meta.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
}

void BoosterMulti::boost_all(std::vector<Histogram>& Hist) {
    meta.reset();
    if (hp.topk == 0) {
        for (size_t i = 0; i < hp.inp_dim; ++i) {
            boost_column_full(Hist[i], i);
        }
    } else {
        if (hp.one_side) {
            for (size_t i = 0; i < hp.inp_dim; ++i) {
                boost_column_topk_one_side(Hist[i], i);
            }
        } else {
            for (size_t i = 0; i < hp.inp_dim; ++i) {
                boost_column_topk_two_side(Hist[i], i);
            }
        }
    }
}

void BoosterMulti::build_tree_best() {
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
        for (size_t i = 0; i < hp.inp_dim; ++i) { Hist_r[i] = Histogram(bin_nums[i], hp.out_dim); }
        hist_all(order_r, Hist_r);
        for (size_t i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_r[i]; }
        Hist_l.assign(info->hist.begin(), info->hist.end());
    } else {
        for (size_t i = 0; i < hp.inp_dim; ++i) { Hist_l[i] = Histogram(bin_nums[i], hp.out_dim); }
        hist_all(order_l, Hist_l);
        for (size_t i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_l[i]; }
        Hist_r.assign(info->hist.begin(), info->hist.end());
    }
    cache.pop_front();

    if (rows_l >= hp.min_samples) {
        if (hp.topk == 0) {
            get_score_opt(Hist_l[rand() % hp.inp_dim], Opt, Score, Score_sum);
        } else {
            get_score_opt(Hist_l[rand() % hp.inp_dim], OptPair, Score, Score_sum);
        }
        boost_all(Hist_l);

        if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
            auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
            tree.add_nonleaf(node, true);
            cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_l, Hist_l));
        } else {
            auto node = LeafNode(hp.out_dim);
            if (hp.topk > 0) { node.Update(parent, OptPair); }
            else { node.Update(parent, Opt); }
            tree.add_leaf(node, true);
        }
    }
    order_l.clear();
    Hist_l.clear();

    if (tree.leaf_num >= hp.max_leaves) { return; }
    if (rows_r >= hp.min_samples) {
        if (hp.topk == 0) {
            get_score_opt(Hist_r[rand() % hp.inp_dim], Opt, Score, Score_sum);
        } else {
            get_score_opt(Hist_r[rand() % hp.inp_dim], OptPair, Score, Score_sum);
        }
        boost_all(Hist_r);
        if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
            auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
            tree.add_nonleaf(node, false);
            cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_r, Hist_r));
        } else {
            auto node = LeafNode(hp.out_dim);
            if (hp.topk > 0) { node.Update(parent, OptPair); }
            else { node.Update(parent, Opt); }
            tree.add_leaf(node, false);
        }
    }
    order_r.clear();
    Hist_r.clear();

    if (!cache.empty()) { build_tree_best(); }
}

void BoosterMulti::update() {
    tree.shrinkage(hp.lr);
    tree.pred_value_multi(Train.Features, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_multi(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(tree);
}

void BoosterMulti::growth() {
    tree.clear();
    cache.clear();

    std::vector<Histogram> Hist(hp.inp_dim);
    for (size_t i = 0; i < hp.inp_dim; ++i) { Hist[i] = Histogram(bin_nums[i], hp.out_dim); }
    hist_all(Train.Orders, Hist);
    if (hp.topk == 0) {
        get_score_opt(Hist[rand() % hp.inp_dim], Opt, Score, Score_sum);
    } else {
        get_score_opt(Hist[rand() % hp.inp_dim], OptPair, Score, Score_sum);
    }
    boost_all(Hist);
    if (meta.column > -1 & meta.gain > -10.0f) {
        auto node = NonLeafNode(-1, meta.column, meta.bin, meta.threshold);
        tree.add_nonleaf(node, true);
        cache.push(CacheInfo(-1, 0, meta, Train.Orders, Hist));
        build_tree_best();
    } else {
        auto node = LeafNode(hp.out_dim);
        if (hp.topk > 0) { node.Update(-1, OptPair); }
        else { node.Update(-1, Opt); }
        tree.add_leaf(node, true);
    }
}

void BoosterMulti::train(int num_rounds) {
    G = malloc_G(Train.num * hp.out_dim);
    H = malloc_H(Train.num * hp.out_dim, obj.constHessian, obj.hessian);

    int round = hp.early_stop == 0 ? num_rounds : hp.early_stop;
    auto early_stoper = EarlyStoper(round, obj.largerBetter);

    // start training
    for (size_t i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, Train.num, hp.out_dim, G, H);
        growth();
        update();
        double score = obj.f_score(Train, Train.num, hp.out_dim);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, hp.out_dim);
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

void BoosterMulti::predict(const double* features, double* preds, const size_t n, int num_trees = 0) {
    num_trees = num_trees == 0 ? int(trees.size()) : std::min(num_trees, int(trees.size()));
    for (size_t i = 0; i < num_trees; ++i) { trees[i].pred_value_multi(features, preds, hp, n); }
}