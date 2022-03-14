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

struct HyperParameters {
    const int inp_dim;
    const int out_dim;
    const char* loss;
    const int max_depth;
    const int max_leaves;
    const int seed;
    const int min_samples;
    const double lr;
    const double reg_l1;
    const double reg_l2;
    const double gamma;
    const double base_score;
    const int early_stop;
    const bool verbose;
    const int max_caches;
    const int topk;
    const bool one_side;
    const int max_bins;
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
