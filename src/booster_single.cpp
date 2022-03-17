#include "booster.h"
#include "histogram.h"
#include "mathFunc.h"

//================================================================================
//                                                                                
//   ####  ##  ##     ##   ####    ##      #####                                
//  ##     ##  ####   ##  ##       ##      ##                                   
//   ###   ##  ##  ## ##  ##  ###  ##      #####                                 
//     ##  ##  ##    ###  ##   ##  ##      ##                                   
//  ####   ##  ##     ##   ####    ######  #####                                
//                                                                                
//================================================================================

BoosterSingle::BoosterSingle(const Shape s, HyperParameters hp) : BoosterBase(s, hp) {};

void BoosterSingle::get_score_opt(Histogram& Hist, double& opt, double& score_sum) {
    double gx = Hist.g[Hist.g.size() - 1];
    double hx = Hist.h[Hist.h.size() - 1];
    const auto CalScore = (hp.reg_l1 == 0) ? CalScoreNoL1 : CalScoreL1;
    const auto CalWeight = (hp.reg_l1 == 0) ? CalWeightNoL1 : CalWeightL1;
    opt = CalWeight(gx, hx, hp.reg_l1, hp.reg_l2);
    score_sum = CalScore(gx, hx, hp.reg_l1, hp.reg_l2);
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

void BoosterSingle::hist_column(
    const std::vector<size_t>& order,
    Histogram& Hist,
    const std::vector<uint16_t>& map_column
) const {
    for (size_t o : order) {
        size_t bin = map_column[o];
        ++Hist.count[bin];
        Hist.g[bin] += this->G[o];
        Hist.h[bin] += this->H[o];
    }
    // Integrate histogram
    for (size_t i = 1; i < Hist.count.size(); ++i) {
        Hist.count[i] += Hist.count[i-1];
        Hist.g[i]  += Hist.g[i-1];
        Hist.h[i]  += Hist.h[i-1];
    }
}

void BoosterSingle::hist_all(
    const std::vector<size_t>& order,
    std::vector<Histogram>& Hist
) const {
    for (size_t i = 0; i < shape.inp_dim; ++i) {
        hist_column(order, Hist[i], Data.train_maps[i]);
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

boost_column_result BoosterSingle::boost_column(const Histogram& Hist, const size_t column) {
    const size_t max_bins = Hist.count.size() - 1;
    const double gx = Hist.g[max_bins];
    const double hx = Hist.h[max_bins];

    boost_column_result result;
    const auto CalScore = (hp.reg_l1 == 0) ? CalScoreNoL1 : CalScoreL1;
    for (size_t i = 0; i < max_bins; ++i) {
        double score_l = CalScore(Hist.g[i], Hist.h[i], hp.reg_l1, hp.reg_l2);
        double score_r = CalScore(gx - Hist.g[i], hx - Hist.h[i], hp.reg_l1, hp.reg_l2);
        double bin_gain = score_l + score_r;
        result.update(bin_gain, i);
    }
    return result;
}

void BoosterSingle::boost_all(const std::vector<Histogram>& Hist) {
    // Boost each column
    std::vector<boost_column_result> results(0);
    for (size_t i = 0; i < shape.inp_dim; ++i) {
        results.push_back(boost_column(Hist[i], i));
    }

    // Update the meta
    meta.reset();
    for (size_t i = 0; i < shape.inp_dim; ++i) {
        auto result = results[i];
        if (result.split_found()) {
            double overall_gain = (result.gain - Score_sum) * 0.5;
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

void BoosterSingle::build_tree_best() {
    if (tree.leaf_num >= hp.max_leaves) { return; }


    auto info = cache.front(); cache.pop_front();
    const int this_node = info.node;
    const size_t rows_l = info.hist[info.split.column].count[info.split.bin];
    const size_t rows_r = info.order.size() - rows_l;

    std::vector<size_t> order_l(rows_l), order_r(rows_r);
    rebuild_order(info.order, order_l, order_r, info.split.column, info.split.bin);
    std::vector<Histogram> Hist_l(shape.inp_dim), Hist_r(shape.inp_dim);

    if (rows_l >= rows_r) {
        for (size_t i = 0; i < shape.inp_dim; ++i) { Hist_r[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_r, Hist_r);
        for (size_t i = 0; i < shape.inp_dim; ++i) { info.hist[i] - Hist_r[i]; }
        Hist_l.assign(info.hist.begin(), info.hist.end());
    } else {
        for (size_t i = 0; i < shape.inp_dim; ++i) { Hist_l[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_l, Hist_l);
        for (size_t i = 0; i < shape.inp_dim; ++i) { info.hist[i] - Hist_l[i]; }
        Hist_r.assign(info.hist.begin(), info.hist.end());
    }

    if (rows_l >= hp.min_samples) {
        get_score_opt(Hist_l[rand() % shape.inp_dim], Opt, Score_sum);
        boost_all(Hist_l);

        if (meta.is_set && info.depth + 1 < hp.max_depth && meta.gain > hp.gamma) {
            tree.add_left_nonleaf(this_node, meta.column, meta.bin, meta.threshold);
            cache.push(CacheInfo(tree.nonleaf_num, info.depth + 1, meta, order_l, Hist_l));
        } else {
            auto node = LeafNode(Opt);
            tree.add_left_leaf(this_node, node);
        }
    }
    order_l.clear();
    Hist_l.clear();

    if (tree.leaf_num >= hp.max_leaves) { return; }

    if (rows_r >= hp.min_samples) {
        get_score_opt(Hist_r[rand() % shape.inp_dim], Opt, Score_sum);
        boost_all(Hist_r);

        if (meta.is_set && info.depth + 1 < hp.max_depth && meta.gain > hp.gamma) {
            tree.add_right_nonleaf(this_node, meta.column, meta.bin, meta.threshold);
            cache.push(CacheInfo(tree.nonleaf_num, info.depth + 1, meta, order_r, Hist_r));
        } else {
            auto node = LeafNode(Opt);
            tree.add_right_leaf(this_node, node);
        }
    }
    order_r.clear();
    Hist_r.clear();

    if (!cache.empty()) { build_tree_best(); }
}

//===========================================================================
//                                                                           
//  ######  #####      ###    ##  ##     ##                                
//    ##    ##  ##    ## ##   ##  ####   ##                                
//    ##    #####    ##   ##  ##  ##  ## ##                                
//    ##    ##  ##   #######  ##  ##    ###                                
//    ##    ##   ##  ##   ##  ##  ##     ##                                
//                                                                           
//===========================================================================

void BoosterSingle::growth() {
    tree.clear();
    cache.clear();

    std::vector<Histogram> Hists(shape.inp_dim);
    for (size_t i = 0; i < shape.inp_dim; ++i) { Hists[i] = Histogram(bin_nums[i], 1); }
    hist_all(Data.train_order, Hists);
    get_score_opt(Hists[rand() % shape.inp_dim], Opt, Score_sum);
    boost_all(Hists);

    // TODO: Parametrise the -10.0??
    //       Also, what is it actually doing here...? Like, what is the meaning
    //       of this parameter? 
    if (meta.is_set && meta.gain > -10.0) {
        tree.add_root_nonleaf(meta.column, meta.bin, meta.threshold);
        cache.push(CacheInfo(-1, 0, meta, Data.train_order, Hists));
        build_tree_best();
    } else {
        auto node = LeafNode(Opt);
        // TODO: (tree.leaf_num >= hp.max_leaves) check ??
        tree.add_left_leaf(-1, node);
    }
    // Finally shrink by the learning rate
    tree.shrink(hp.learning_rate);
}

void BoosterSingle::train(int num_rounds) {
    srand(hp.seed);
    G = malloc_G();
    H = malloc_H(obj.constHessian, obj.hessian);
    auto early_stoper = EarlyStopper(hp.early_stop == 0 ? num_rounds : hp.early_stop, obj.largerBetter);

    const int out_dim = shape.out_dim;

    // start training
    for (size_t i = 0; i < num_rounds; ++i) {
        // Calculate the gradeint and the hessian for all the trees (columns of )
        obj.f_grad(Data, out_dim, G, H);
        for (size_t j = 0; j < out_dim; ++j) {
            growth();

            // for (int j = 0; j < 30; j++){ std::cout << Data.Features[j] << " "; }; std::cout << "\n";
            tree.pred_value_single(Data.Features, Data.preds, shape, Data.n);
            // for (int j = 0; j < 30; j++){ std::cout << Data.preds[j] << " "; }; std::cout << "\n";
            trees.push_back(tree);

            if (j < out_dim - 1) {
                G += Data.n;
                H += Data.n;
                Data.preds += Data.n;
            } else {
                const int pos = (out_dim - 1) * Data.n;
                G -= pos;
                H -= pos;
                Data.preds -= pos;
            }
        }
        double train_score = obj.f_partial_score(Data, out_dim, G, true);
        if (hp.eval_fraction > 0.0) {
            // double eval_score = obj.f_partial_score(Data, out_dim, G, false);
            double metric = obj.f_metric(Data, out_dim, G, false);
            if (hp.verbose) { showloss(train_score, metric, i); }
            early_stoper.push(std::make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.info;
                int round = std::get<1>(info);
                showbest(std::get<0>(info), round);
                trees.resize(shape.out_dim * round);
                break;
            }
        } else {
            if (hp.verbose) { showloss(train_score, i); }
        }

    }
    if (early_stoper.is_continue && hp.eval_fraction > 0.0) { showbest(early_stoper.info); }
    free(G);
    free(H);
}

//=====================================================================================
//                                                                                     
//  #####   #####    #####  ####    ##   ####  ######                                
//  ##  ##  ##  ##   ##     ##  ##  ##  ##       ##                                  
//  #####   #####    #####  ##  ##  ##  ##       ##                                  
//  ##      ##  ##   ##     ##  ##  ##  ##       ##                                  
//  ##      ##   ##  #####  ####    ##   ####    ##                                  
//                                                                                     
//=====================================================================================

void BoosterSingle::predict(
    const double* const features,
    double* const preds,
    const size_t n, 
    size_t num_trees = 0
) {
    size_t max_trees = trees.size() / shape.out_dim;
    num_trees = (num_trees == 0) ? max_trees : std::min(num_trees, max_trees);
    for (size_t i = 0; i < num_trees; ++i) {
        for (size_t j = 0; j < shape.out_dim; ++j) {
            trees[(i*shape.out_dim)+j].pred_value_single(features, preds + (j*n), shape, n);
        }
    }
}