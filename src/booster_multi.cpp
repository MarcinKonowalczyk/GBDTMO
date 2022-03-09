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

//=============================================================================================
//                                                                                             
//  ##   ##  ##   ####  ######          ###    ##      ##                                    
//  ##   ##  ##  ##       ##           ## ##   ##      ##                                    
//  #######  ##   ###     ##          ##   ##  ##      ##                                    
//  ##   ##  ##     ##    ##          #######  ##      ##                                    
//  ##   ##  ##  ####     ##          ##   ##  ######  ######                                
//                                                                                             
//=============================================================================================

void BoosterMulti::hist_column_multi(
    const std::vector<size_t>& order,
    Histogram& Hist,
    const uint16_t* maps
) {
    size_t out_dim = hp.out_dim;
    for (size_t i : order) {
        ++Hist.count[maps[i]];
        size_t bin = maps[i] * out_dim;
        size_t ind = i * out_dim;
        for (size_t j = 0; j < out_dim; ++j) {
            Hist.g[bin+j] += G[ind+j];
            Hist.h[bin+j] += H[ind+j];
        }
    }
    // integration
    size_t ind = 0;
    for (size_t i = 1; i < Hist.count.size(); ++i) {
        Hist.count[i] += Hist.count[i - 1];
        for (size_t j = 0; j < out_dim; ++j) {
            Hist.g[ind + out_dim] += Hist.g[ind];
            Hist.h[ind + out_dim] += Hist.h[ind];
            ++ind;
        }
    }
}

void BoosterMulti::hist_all(
    const std::vector<size_t>& order,
    std::vector<Histogram>& Hist
) {
    for (size_t i = 0; i < hp.inp_dim; ++i) {
        hist_column_multi(order, Hist[i], Train.Maps + i*Train.num);
    }
}

//==========================================================================================================
//                                                                                                          
//  #####    #####    #####    ####  ######          ###    ##      ##                                    
//  ##  ##  ##   ##  ##   ##  ##       ##           ## ##   ##      ##                                    
//  #####   ##   ##  ##   ##   ###     ##          ##   ##  ##      ##                                    
//  ##  ##  ##   ##  ##   ##     ##    ##          #######  ##      ##                                    
//  #####    #####    #####   ####     ##          ##   ##  ######  ######                                
//                                                                                                          
//==========================================================================================================


boost_column_result
BoosterMulti::boost_column_full(const Histogram& Hist, const size_t column) {
    const double* gx = &Hist.g[Hist.g.size() - hp.out_dim];
    const double* hx = &Hist.h[Hist.h.size() - hp.out_dim];
    const size_t max_bins = Hist.count.size() - 1; // TODO: Make sure Hist.count.size > 0 ?

    boost_column_result result;
    for (size_t i = 0; i < max_bins; ++i) {
        size_t bo = i * hp.out_dim; // bin offset
        
        // Gain for all the output variables
        double bin_gain = 0.0;
        for (size_t j = 0; j < hp.out_dim; ++j) {
            size_t ho = bo + j; // histogram offset
            double score_l = CalScore(Hist.g[ho], Hist.h[ho], hp.reg_l1, hp.reg_l2);
            double score_r = CalScore(gx[j] - Hist.g[ho], hx[j] - Hist.h[ho], hp.reg_l1, hp.reg_l2);
            bin_gain += score_l + score_r;
        }
        result.update(bin_gain, i);
    }
    return result;
}

boost_column_result
BoosterMulti::boost_column_topk_two_side(const Histogram& Hist, const size_t column) {
    const double* gx = &Hist.g[Hist.g.size() - hp.out_dim];
    const double* hx = &Hist.h[Hist.h.size() - hp.out_dim];
    const size_t max_bins = Hist.count.size() - 1;

    auto pq_l = TopkPriority<double>(hp.topk);
    auto pq_r = TopkPriority<double>(hp.topk);

    boost_column_result result;
    for (size_t i = 0; i < max_bins; ++i) {
        size_t bo = i * hp.out_dim; // bin offset

        // Go through all the splits
        for (size_t j = 0; j < hp.out_dim; ++j) {
            size_t ho = bo + j; // histogram offset
            double score_l = CalScore(Hist.g[ho], Hist.h[ho], hp.reg_l1, hp.reg_l2);
            double score_r = CalScore(gx[j] - Hist.g[ho], hx[j] - Hist.h[ho], hp.reg_l1, hp.reg_l2);
            pq_l.push(score_l);
            pq_r.push(score_r);
        }

        double bin_gain = 0.0;
        for(; !pq_l.empty(); pq_l.pop()) { bin_gain += pq_l.top(); }
        for(; !pq_r.empty(); pq_r.pop()) { bin_gain += pq_r.top(); }
        result.update(bin_gain, i);
    }
    return result;
}

boost_column_result
BoosterMulti::boost_column_topk_one_side(const Histogram& Hist, const size_t column) {
    const double* gx = &Hist.g[Hist.g.size() - hp.out_dim];
    const double* hx = &Hist.h[Hist.h.size() - hp.out_dim];
    const size_t max_bins = Hist.count.size() - 1;

    auto pq = TopkPriority<double>(hp.topk);

    boost_column_result result;
    for (size_t i = 0; i < max_bins; ++i) {
        size_t bo = i * hp.out_dim; // bin offset
        
        for (size_t j = 0; j < hp.out_dim; ++j) {
            size_t ho = bo + j; // histogram offset
            double score_l = CalScore(Hist.g[ho], Hist.h[ho], hp.reg_l1, hp.reg_l2);
            double score_r = CalScore(gx[j] - Hist.g[ho], hx[j] - Hist.h[ho], hp.reg_l1, hp.reg_l2);
            pq.push(score_l + score_r);
        }

        double bin_gain = 0.0;
        for (; !pq.empty(); pq.pop()) { bin_gain += pq.top(); }
        result.update(bin_gain, i);
    }
    return result;
}

void BoosterMulti::boost_all(const std::vector<Histogram>& Hist) {
    std::vector<boost_column_result> results;
    results.reserve(hp.inp_dim);
    if (hp.topk == 0) {
        for (size_t i = 0; i < hp.inp_dim; ++i) {
            results.push_back(boost_column_full(Hist[i], i));
        }
    } else {
        if (hp.one_side) {
            for (size_t i = 0; i < hp.inp_dim; ++i) {
                results.push_back(boost_column_topk_one_side(Hist[i], i));
            }
        } else {
            for (size_t i = 0; i < hp.inp_dim; ++i) {
                results.push_back(boost_column_topk_two_side(Hist[i], i));
            }
        }
    }

    int divisor = (hp.topk == 0) ? hp.out_dim : hp.topk;
    meta.reset();
    for (size_t i  = 0; i < hp.inp_dim; ++i) {
        auto result = results[i];
        if (result.split_found()) {
            double overall_gain =  (result.gain - Score_sum) / (2 * divisor);
            if (overall_gain > meta.gain) {
                meta.update(overall_gain, i, result.bin_index, bin_values[i][result.bin_index]);
            }
        }
    }
}

//============================================================================================================
//                                                                                                            
//  #####   ##   ##  ##  ##      ####          ######  #####    #####  #####                                
//  ##  ##  ##   ##  ##  ##      ##  ##          ##    ##  ##   ##     ##                                   
//  #####   ##   ##  ##  ##      ##  ##          ##    #####    #####  #####                                
//  ##  ##  ##   ##  ##  ##      ##  ##          ##    ##  ##   ##     ##                                   
//  #####    #####   ##  ######  ####            ##    ##   ##  #####  #####                                
//                                                                                                            
//============================================================================================================

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
        if (meta.isset && depth + 1 < hp.max_depth && meta.gain > hp.gamma) {
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
        if (meta.isset && depth + 1 < hp.max_depth && meta.gain > hp.gamma) {
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
    if (meta.isset && meta.gain > -10.0f) {
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