#ifndef GBDTMO_DATASTRUCTURE_H
#define GBDTMO_DATASTRUCTURE_H

#include <math.h>
#include <queue>
#include <iostream>
#include <random>
#include <utility>
#include <limits.h>

struct Dataset {
    int num = 0;
    double* Features;
    uint16_t* Maps;
    double* Preds;
    std::vector<int32_t> Orders;
    std::vector<int32_t> LeafIndex;
    double* Label_double;
    int32_t* Label_int32;
};

struct HyperParameter {
    const int inp_dim;
    const int out_dim;
    const char* loss; // = "mse";
    const int max_depth; // = 5;
    const int max_leaves; // = int(pow(2, max_depth));
    const int seed; // = 0; // unsigned ?
    const int min_samples; // = 2; // 5
    const double lr; // = 0.2;
    const double reg_l1; // = 0.0;
    const double reg_l2; // = 1.0;
    const double gamma; // = 1e-3;
    const double base_score; // = 0.0f;
    const int early_stop; // = 0;
    const bool verbose; // = true;
    const int max_caches; // = 16; // hist_cache
    const int topk; // = 0;
    const bool one_side; // = true;
};
// const bool hist; // = true;
// const bool best_first; // = true;
// const double max_delta; // = 10.0;

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

//
class EarlyStoper {
private:
    int k = 10;
    bool larger_better = true;

public:
    bool is_continue = true;
    std::pair<double, int> info;

    EarlyStoper() {};

    EarlyStoper(int n, bool lb) : k(n), larger_better(lb) {
        info = std::make_pair(lb ? -1e10 : 1e10, 0);
    };

    inline void push(const std::pair<double, int>& x) {
        if (larger_better == x > info) { info = x; } // xnor(larger_better, x > info)
        is_continue = (std::get<1>(x) < std::get<1>(info) + k);
    }
};


#endif /* GBDTMO_DATASTRUCTURE_H */
