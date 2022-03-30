#ifndef GBDTMO_DATASTRUCTURE_H
#define GBDTMO_DATASTRUCTURE_H

#include <math.h>
#include <queue>
#include <iostream>
#include <utility>

// TODO: ?? typedef Maps uint16_t*

struct Shape {
    size_t inp_dim;
    size_t out_dim;

    Shape(size_t i, size_t o) : inp_dim(i), out_dim(o) {};

    inline void show() const {
        std::cout << "Shape = [\n";
        std::cout << " .inp_dim = " << inp_dim << "\n";
        std::cout << " .out_dim = " << out_dim << "\n";
        std::cout << "]\n";
    };

};

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
    size_t n = 0;
    float* Features; // N x inp_dim
    float* preds; // N x out_dim
    float* Label_float; // N
    int32_t* Label_int32; // N
    std::vector<size_t> train_order; // N_train
    std::vector<size_t> eval_order; // N_eval
    std::vector<std::vector<uint16_t>> train_maps; // N_train x inp_dim
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
    Loss loss;
    unsigned int max_depth;
    unsigned int max_leaves;
    int seed;
    unsigned int min_samples;
    float learning_rate;
    float reg_l1;
    float reg_l2;
    float gamma;
    unsigned int early_stop;
    bool verbose;
    unsigned int max_caches;
    unsigned int topk;
    bool one_side;
    unsigned int max_bins;
    float alpha;
    float eval_fraction;


    void init_default() {
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

    inline void show() const {
        std::cout << "HyperParameters = [...\n";
        // std::cout << " .loss = " << loss << "\n";
        std::cout << " .max_depth = " << max_depth << "\n";
        // std::cout << " .max_leaves = " << max_leaves << "\n";
        // std::cout << " .seed = " << seed << "\n";
        // std::cout << " .min_samples = " << min_samples << "\n";
        std::cout << " .learning_rate = " << learning_rate << "\n";
        std::cout << " .reg_l1 = " << reg_l1 << "\n";
        std::cout << " .reg_l2 = " << reg_l2 << "\n";
        // std::cout << " .gamma = " << gamma << "\n";
        // std::cout << " .early_stop = " << early_stop << "\n";
        // std::cout << " .verbose = " << verbose << "\n";
        // std::cout << " .max_caches = " << max_caches << "\n";
        std::cout << " .topk = " << topk << "\n";
        // std::cout << " .one_side = " << one_side << "\n";
        // std::cout << " .max_bins = " << max_bins << "\n";
        // std::cout << " .alpha = " << alpha << "\n";
        // std::cout << " .eval_fraction = " << eval_fraction << "\n";
        std::cout << "]\n";
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
    inline int search(T const& x) {
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

    inline void push(T const& x) {
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

    inline void push(T const& x) {
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
    std::pair<float, int> info;

    EarlyStopper() {};

    EarlyStopper(int n, bool lb) : k(n), larger_better(lb) {
        info = std::make_pair(lb ? -1e10 : 1e10, 0);
    };

    inline void stop() { is_continue = false; }

    inline void push(const std::pair<float, int>& x) {
        if (larger_better == x > info) { info = x; } // xnor(larger_better, x > info)
        is_continue = is_continue && (std::get<1>(x) < std::get<1>(info) + k);
    }
};


#endif /* GBDTMO_DATASTRUCTURE_H */
