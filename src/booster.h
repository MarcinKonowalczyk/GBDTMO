#ifndef GBDTMO_BOOSTER_H
#define GBDTMO_BOOSTER_H

#include "tree.h"
#include "mathFunc.h"
#include "datastruct.h"
#include "loss.h"
#include "io.h"
#include "histogram.h"
// #include <algorithm>

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

    bool operator>(const CacheInfo& x) const { return split.gain > x.split.gain; }
    
};

struct boost_column_result {
    float gain = 0.0;
    size_t bin_index = 0;

    inline bool split_found() { return gain > 0.0; };

    inline void update(float new_gain, size_t index) {
        if (new_gain > gain) {
            gain = new_gain;
            bin_index = index;
        }
    }

};

//=================================================================
//                                                                 
//  #####     ###     ####  #####                                
//  ##  ##   ## ##   ##     ##                                   
//  #####   ##   ##   ###   #####                                
//  ##  ##  #######     ##  ##                                   
//  #####   ##   ##  ####   #####                                
//                                                                 
//=================================================================

class BoosterBase {
public:
    BoosterBase(const Shape, HyperParameters);
    virtual ~BoosterBase() = default;
    
    void sigint_handler(int signum);

    void set_gh(float*, float*);
    void set_data(float* features, float* preds, size_t n);
    void set_label(float*);
    void set_label(int32_t*);
    void calc_maps();
    void calc_eval_fraction();

    void rebuild_order(
        const std::vector<size_t>& order,
        std::vector<size_t>& order_l,
        std::vector<size_t>& order_r,
        const size_t split_column,
        const uint16_t bin
    ) const;

    // Print helpers
    inline void showloss(float score, float metric, int i) const { std::cout << "[" << i << "] train->" << std::setprecision(5) << std::fixed << score << "\teval->" << std::setprecision(5) << std::fixed << metric << std::endl; }
    inline void showloss(float metric, int i) const { std::cout << "[" << i << "] score->" << std::setprecision(5) << std::fixed << metric << std::endl; }
    inline void showbest(std::pair<float, int> info) const { showbest(std::get<0>(info), std::get<1>(info)); }
    inline void showbest(float score, int round) const { std::cout << "Best score " << score << " at round " << round << std::endl; }

    // IO
    void load(const char* path) { LoadTrees(trees, path); }
    void dump(const char* path) { DumpTrees(trees, path); }

    // API
    virtual void growth() = 0;
    virtual void train(int) = 0;
    virtual void predict(
        const float* const features,
        float* const preds,
        const size_t n,
        size_t num_trees // NOTE: num_trees can change inside of the function. If num_trees = 0, num_trees == max_trees
    ) = 0;
    void reset();
    std::vector<Tree> trees;

    // Used to dump state to arrays
    void dump_nonleaf_sizes(uint16_t* nonleaf_sizes) const;
    void dump_leaf_sizes(uint16_t* leaf_sizes) const;
    void dump_nonleaf_nodes(int* trees, float* thresholds) const;
    void dump_leaf_nodes(float* leaves) const;
    
    const Shape shape;
    HyperParameters hp;

protected:

    float* malloc_G() const;
    float* malloc_H(const bool constHessian, const float constValue) const;

    virtual void boost_all(const std::vector<Histogram>& Hist) = 0;
    virtual void hist_column(
        const std::vector<size_t>& order,
        Histogram& Hist,
        const std::vector<uint16_t>& map_column
    ) const = 0;
    virtual void hist_all(
        const std::vector<size_t>& order,
        std::vector<Histogram>& Hist
    ) const = 0;

    Tree tree;
    std::vector<uint16_t> bin_nums;
    std::vector<std::vector<float>> bin_values;
    SplitInfo meta;
    Dataset Data;
    std::vector<size_t> eval_indices;
    std::vector<size_t> train_indices;
    float* G;
    float* H;
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
    BoosterSingle(const Shape, HyperParameters);
    void growth() override;
    void train(int) override;
    void predict(
        const float* const features,
        float* const preds,
        const size_t n,
        size_t num_trees // NOTE: num_trees can change inside of the function. If num_trees = 0, num_trees == max_trees
    ) override;
    void reset();

private:

    boost_column_result boost_column(const Histogram& Hist, const size_t column);
    void boost_all(const std::vector<Histogram>& Hist) override;

    void hist_column(
        const std::vector<size_t>& order,
        Histogram& Hist,
        const std::vector<uint16_t>& map_column
    ) const override;
    void hist_all(
        const std::vector<size_t>& order,
        std::vector<Histogram>& Hist
    ) const override;

    float Score_sum, Opt;
    void get_score_opt(Histogram&, float& , float& );
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
    BoosterMulti(const Shape, const HyperParameters);
    void growth() override;
    void train(int) override;
    void predict(
        const float* const features,
        float* const preds,
        const size_t n,
        size_t num_trees // NOTE: num_trees can change inside of the function. If num_trees = 0, num_trees == max_trees
    ) override;

private:
    
    boost_column_result boost_column_full(const Histogram& Hist, const size_t column);
    boost_column_result boost_column_topk_two_side(const Histogram& Hist, const size_t column);
    boost_column_result boost_column_topk_one_side(const Histogram& Hist, const size_t column);
    void boost_all(const std::vector<Histogram>& Hist) override;

    // void hist_column_multi(const std::vector<size_t>& order, Histogram& Hist, const uint16_t* maps) const;
    void hist_column(
        const std::vector<size_t>& order,
        Histogram& Hist,
        const std::vector<uint16_t>& map_column
    ) const override;
    void hist_all(
        const std::vector<size_t>& order,
        std::vector<Histogram>& Hist
    ) const override;

    float Score_sum;
    std::vector<float> Score;
    std::vector<float> Opt;
    std::vector<std::pair<float, int>> OptPair;

    void get_score_opt(
        const Histogram& Hist,
        std::vector<float>& opt,
        std::vector<float>& score,
        float& score_sum
    ) const;

    void get_score_opt(
        const Histogram& Hist,
        std::vector<std::pair<float, int>>& opt,
        std::vector<float>& score,
        float& score_sum
    ) const;

    void build_tree_best();
};

#endif /* GBDTMO_BOOSTER_H */