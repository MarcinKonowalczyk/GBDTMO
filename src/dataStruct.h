#ifndef GBDTMO_DATASTRUCTURE_H
#define GBDTMO_DATASTRUCTURE_H

#include <math.h>
#include <queue>
#include <iostream>
#include <random>
#include <utility>
#include <limits.h>

struct Histogram {
    Histogram(int n = 1, int d = 1) : count(n, 0), g(n * d, 0), h(n * d, 0) {}

    std::vector<int> count;
    std::vector<double> g;
    std::vector<double> h;

    inline void operator-(const Histogram &x) {

        for (int i = 0; i < x.count.size(); ++i) {
            count[i] -= x.count[i];
        }
        for (int i = 0; i < x.g.size(); ++i) {
            g[i] -= x.g[i];
            h[i] -= x.h[i];
        }
    }
};

void histogram_single(std::vector<int32_t> &, Histogram &, uint16_t *, double *, double *);

void histogram_multi(std::vector<int32_t> &, Histogram &, uint16_t *, double *, double *, int);

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
    bool hist = true;
    bool one_side = true;
    bool best_first = true;
    bool verbose = true;
    const char *loss = "mse";
    int max_depth = 5;
    int max_leaves = int(pow(2, max_depth));
    int inp_dim;
    int out_dim = 1;
    int min_samples = 2;
    double lr = 0.2;
    double reg_l1 = 0.0;
    double reg_l2 = 1.0;
    double gamma = 1e-3;
    double max_delta = 10.0;
    unsigned int seed = 0;
    double base_score = 0.0f;
    int topk = 0;
    int Max_caches = 16;
    int early_stop = 0;
};

// access to all top-k elements
template<class T>
class TopkDeque {
private:
    int max_size;

public:
    std::deque<T> data;

    TopkDeque(int n = 8) : max_size(n) {};

    // binary search
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

    EarlyStoper(int n, bool state) : k(n), larger_better(state) {
        info = std::make_pair(state ? -1e10 : 1e10, 0);
    };

    inline void push(const std::pair<double, int> &x) {
        if (larger_better == x > info) { info = x; } // xnor(larger_better, x > info)
        is_continue = (std::get<1>(x) < std::get<1>(info) + k);
    }
};


#endif /* GBDTMO_DATASTRUCTURE_H */
