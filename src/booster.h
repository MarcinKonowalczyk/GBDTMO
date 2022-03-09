#ifndef GBDTMO_BOOSTER_H
#define GBDTMO_BOOSTER_H

#include "tree.h"
#include "mathFunc.h"
#include "datastruct.h"
#include "loss.h"
#include "io.h"
#include "histogram.h"
#include <algorithm>
// #include <cstdlib> /* malloc */

struct CacheInfo {
    int node;
    int depth;
    SplitInfo split;
    std::vector<size_t> order;
    std::vector<Histogram> hist;

    CacheInfo(
        int n,
        int d,
        SplitInfo& s,
        std::vector<size_t>& o,
        std::vector<Histogram>& h
    ) : node(n), depth(d), split(s), order(o), hist(h) {};

    bool operator>(const CacheInfo &x) const { return split.gain > x.split.gain; }
};

//======================================================================
//                                                                      
//  ##   ##  ######  ##  ##       ####                                
//  ##   ##    ##    ##  ##      ##                                   
//  ##   ##    ##    ##  ##       ###                                 
//  ##   ##    ##    ##  ##         ##                                
//   #####     ##    ##  ######  ####                                 
//                                                                      
//======================================================================


class BoosterBase {
public:
    BoosterBase(const HyperParameters p);

    void set_gh(double*, double*);
    void set_train_data(double* features, double* preds, int n);
    void set_eval_data(double* features, double* preds, int n);
    void set_train_label(double*);
    void set_train_label(int32_t*);
    void set_eval_label(double*);
    void set_eval_label(int32_t*);

    void calc_train_maps();

    void rebuild_order(
        const std::vector<size_t>& order,
        std::vector<size_t>& order_l,
        std::vector<size_t>& order_r,
        const uint16_t* maps,
        const uint16_t bin
    );

    double* malloc_G(int elements);
    double* malloc_H(int elements, bool constHessian, double constValue);

    // Print helpers
    inline void showloss(double score, double metric, int i) const { std::cout << "[" << i << "] train->" << std::setprecision(5) << std::fixed << score << "\teval->" << std::setprecision(5) << std::fixed << metric << std::endl; }
    inline void showloss(double metric, int i) const { std::cout << "[" << i << "] score->" << std::setprecision(5) << std::fixed << metric << std::endl; }
    inline void showbest(std::pair<double, int> info) const { showbest(std::get<0>(info), std::get<1>(info)); }
    inline void showbest(double score, int round) const { std::cout << "Best score " << score << " at round " << round << std::endl; }

    // IO
    void load(const char* path) { LoadTrees(trees, path); }
    void dump(const char* path) { DumpTrees(trees, path); }

    // API
    virtual void update() = 0;
    virtual void growth() = 0;
    virtual void train(int) = 0;
    virtual void predict(const double* features, double* preds, const size_t n, int num_trees) = 0;
    void reset();

protected:
    virtual void boost_all(const std::vector<Histogram>& Hist) = 0;
    virtual void hist_all(std::vector<size_t>& order, std::vector<Histogram>& Hist) = 0;

    Tree tree;
    std::vector<Tree> trees;
    std::vector<uint16_t> bin_nums;
    std::vector<std::vector<double>> bin_values;
    SplitInfo meta;
    Dataset Train;
    Dataset Eval;
    double* G;
    double* H;
    const HyperParameters hp;
    TopkDeque<CacheInfo> cache;
    Objective obj;
};

//================================================================================
//                                                                                
//   ####  ##  ##     ##   ####    ##      #####                                
//  ##     ##  ####   ##  ##       ##      ##                                   
//   ###   ##  ##  ## ##  ##  ###  ##      #####                                
//     ##  ##  ##    ###  ##   ##  ##      ##                                   
//  ####   ##  ##     ##   ####    ######  #####                                
//                                                                                
//================================================================================

class BoosterSingle : public BoosterBase {
public:
    BoosterSingle(const HyperParameters hp);
    void update() override;
    void growth() override;
    void train(int) override;
    void predict(const double* features, double* preds, const size_t n, int num_trees) override;
    void reset();
    void predict_multi(const double* features, double* preds, const size_t n, const int out_dim, int num_trees);

private:

    void boost_column(const Histogram& Hist, const size_t column);
    void boost_all(const std::vector<Histogram>& Hist) override;

    void hist_all(std::vector<size_t>& order, std::vector<Histogram>& Hist) override;

    double Score_sum, Opt;
    void get_score_opt(Histogram&, double& , double& );
    void build_tree_best();
};

//===========================================================================
//                                                                           
//  ###    ###  ##   ##  ##      ######  ##                                
//  ## #  # ##  ##   ##  ##        ##    ##                                
//  ##  ##  ##  ##   ##  ##        ##    ##                                
//  ##      ##  ##   ##  ##        ##    ##                                
//  ##      ##   #####   ######    ##    ##                                
//                                                                           
//===========================================================================

class BoosterMulti : public BoosterBase {
public:
    BoosterMulti(const HyperParameters hp);
    void update() override;
    void growth() override;
    void train(int) override;
    void predict(const double* features, double* preds, const size_t n, int num_trees) override;

private:
    
    void boost_column_full(const Histogram& Hist, const size_t column);
    void boost_column_topk_two_side(const Histogram& Hist, const size_t column);
    void boost_column_topk_one_side(const Histogram& Hist, const size_t column);
    void boost_all(const std::vector<Histogram>& Hist) override;

    void hist_all(std::vector<size_t>& order, std::vector<Histogram>& Hist) override;

    double Score_sum;
    std::vector<double> Score;
    std::vector<double> Opt;
    std::vector<std::pair<double, int>> OptPair;
    void get_score_opt(Histogram&, std::vector<double>&, std::vector<double>&, double&);
    void get_score_opt(Histogram&, std::vector<std::pair<double, int>>& , std::vector<double>&, double&);
    void build_tree_best();
};

#endif /* GBDTMO_BOOSTER_H */