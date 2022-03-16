#ifndef GBDTMO_DATASTRUCTURE_H
#define GBDTMO_DATASTRUCTURE_H

#include <math.h>
#include <queue>
#include <iostream>
#include <random>
#include <utility>

// #include <limits.h>

// TODO: ?? typedef Maps uint16_t*

//==========================================================================================
//                                                                                          
//  ####      ###    ######    ###     ####  #####  ######                                
//  ##  ##   ## ##     ##     ## ##   ##     ##       ##                                  
//  ##  ##  ##   ##    ##    ##   ##   ###   #####    ##                                  
//  ##  ##  #######    ##    #######     ##  ##       ##                                  
//  ####    ##   ##    ##    ##   ##  ####   #####    ##                                  
//                                                                                          
//==========================================================================================

struct Dataset {
    int num = 0;
    double* Features;
    uint16_t* Maps;
    double* Preds;
    std::vector<size_t> Orders;
    std::vector<size_t> LeafIndex;
    double* Label_double;
    int32_t* Label_int32;
};

//===================================================
//                                                   
//  ##   ##  #####                                 
//  ##   ##  ##  ##                                
//  #######  #####                                 
//  ##   ##  ##                                    
//  ##   ##  ##                                    
//                                                   
//===================================================

enum Loss { mse, ce, ce_column, bce };

constexpr static unsigned int hash(const char* s, int off = 0) {                        
    return !s[off] ? 5381 : (hash(s, off+1)*33) ^ s[off];                           
} 

struct HyperParameters {
    int inp_dim;
    int out_dim;
    Loss loss;
    int max_depth;
    int max_leaves;
    int seed;
    int min_samples;
    double learning_rate;
    double reg_l1;
    double reg_l2;
    double gamma;
    int early_stop;
    bool verbose;
    int max_caches;
    int topk;
    bool one_side;
    int max_bins;
    double alpha;
    double eval_fraction;


    void init_default() {
        inp_dim = 1;
        out_dim = 1;
        loss = Loss::mse;
        max_depth = 5;
        max_leaves = 32;
        seed = 0;
        min_samples = 5;
        learning_rate = 0.1;
        reg_l1 = 0.0;
        reg_l2 = 0.1;
        gamma = 1e-3;
        early_stop = 0;
        verbose = true;
        max_caches = 16;
        topk = 0;
        one_side = true;
        max_bins = 32;
        alpha = 0.0;
        eval_fraction = 0.0;
    }
};

//=================================================================================================
//                                                                                                 
//  ######   #####   #####   ##  ##        ####    #####   #####                                 
//    ##    ##   ##  ##  ##  ## ##         ##  ##  ##     ##   ##                                
//    ##    ##   ##  #####   ####          ##  ##  #####  ##   ##                                
//    ##    ##   ##  ##      ## ##         ##  ##  ##      #####                                 
//    ##     #####   ##      ##  ##        ####    #####  ##                                     
//                                                                                                 
//=================================================================================================

// access to all top-k elements
template<class T>
class TopkDeque {
private:
    int max_size;

public:
    std::deque<T> data;

    TopkDeque(int n = 8) : max_size(n) {};

    //binary search
    inline int search(T const &x) {
        int l = 0, r = data.size();
        while (l < r) {
            int m = (r + l - 1) / 2;
            if (x > data[m]) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }

    inline void push(T const &x) {
        if (data.size() == max_size) {
            if (x > data[max_size - 1]) {
                int i = search(x);
                data.insert(data.begin() + i, x);
                data.pop_back();
            }
        } else {
            int i = search(x);
            data.insert(data.begin() + i, x);
        }
    }

    inline bool empty() { return data.empty(); }
    inline void clear() { data.clear(); }
    inline void pop_front() { data.pop_front(); }
    inline T front() { return data.front(); }
};

//================================================================================================================
//                                                                                                                
//  ######   #####   #####   ##  ##        #####   #####    ##   #####   #####                                  
//    ##    ##   ##  ##  ##  ## ##         ##  ##  ##  ##   ##  ##   ##  ##  ##                                 
//    ##    ##   ##  #####   ####          #####   #####    ##  ##   ##  #####                                  
//    ##    ##   ##  ##      ## ##         ##      ##  ##   ##  ##   ##  ##  ##                                 
//    ##     #####   ##      ##  ##        ##      ##   ##  ##   #####   ##   ##                                
//                                                                                                                
//================================================================================================================

// Priority queue with capacity of k elements and the smallest element on top
template<class T>
class TopkPriority {
private:
    int k = 8;

public:
    std::priority_queue<T, std::vector<T>, std::greater<T>> data;

    TopkPriority(int n) : k(n) {};

    inline void push(T const &x) {
        data.push(x);
        if (data.size() > k) {
            data.pop();
        }
    }

    inline void pop() { data.pop(); }
    inline bool empty() { return data.empty(); }
    inline T top() { return data.top(); }
};

//===========================================================================================================================================
//                                                                                                                                           
//  #####    ###    #####    ##      ##    ##         ####  ######   #####   #####   #####   #####  #####                                  
//  ##      ## ##   ##  ##   ##       ##  ##         ##       ##    ##   ##  ##  ##  ##  ##  ##     ##  ##                                 
//  #####  ##   ##  #####    ##        ####           ###     ##    ##   ##  #####   #####   #####  #####                                  
//  ##     #######  ##  ##   ##         ##              ##    ##    ##   ##  ##      ##      ##     ##  ##                                 
//  #####  ##   ##  ##   ##  ######     ##           ####     ##     #####   ##      ##      #####  ##   ##                                
//                                                                                                                                           
//===========================================================================================================================================

class EarlyStopper {
private:
    int k = 10;
    bool larger_better = true;

public:
    bool is_continue = true;
    std::pair<double, int> info;

    EarlyStopper() {};

    EarlyStopper(int n, bool lb) : k(n), larger_better(lb) {
        info = std::make_pair(lb ? -1e10 : 1e10, 0);
    };

    inline void push(const std::pair<double, int>& x) {
        if (larger_better == x > info) { info = x; } // xnor(larger_better, x > info)
        is_continue = (std::get<1>(x) < std::get<1>(info) + k);
    }
};


#endif /* GBDTMO_DATASTRUCTURE_H */
