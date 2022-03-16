#ifndef GBDTMO_MATH_H
#define GBDTMO_MATH_H

#include <vector>
#include <math.h>
#include <unordered_set>

template<typename T>
inline static T Sum(std::vector<T> x) {
    T s = 0;
    for (auto& i : x) { s += i; }
    return s;
}

template<typename T>
inline static T Sqr(T x) { return x * x; }

inline static double ThresholdL1(double w, double reg_l1) {
    if (w > +reg_l1) return w - reg_l1;
    if (w < -reg_l1) return w + reg_l1;
    return 0.0;
}

//===================================================================================
//                                                                                   
//  ##      ##  #####  ##   ####    ##   ##  ######                                
//  ##      ##  ##     ##  ##       ##   ##    ##                                  
//  ##  ##  ##  #####  ##  ##  ###  #######    ##                                  
//  ##  ##  ##  ##     ##  ##   ##  ##   ##    ##                                  
//   ###  ###   #####  ##   ####    ##   ##    ##                                  
//                                                                                   
//===================================================================================

inline static double CalWeightNoL1(
    const double g_sum,
    const double h_sum,
    const double reg_l1,
    const double reg_l2
) {
    return -g_sum / (h_sum + reg_l2);
}

inline static double CalWeightL1(
    const double g_sum,
    const double h_sum,
    const double reg_l1,
    const double reg_l2
) {
    return -ThresholdL1(g_sum, reg_l1) / (h_sum + reg_l2);
}


// inline static void CalWeight(
//     std::vector<double>& value,
//     const double* g_sum,
//     const double* h_sum,
//     const double reg_l1,
//     const double reg_l2
// ) {
//     if (reg_l1 == 0) {
//         for (int i = 0; i < value.size(); ++i) { value[i] = -g_sum[i] / (h_sum[i] + reg_l2); }
//     } else {
//         for (int i = 0; i < value.size(); ++i) { value[i] = -ThresholdL1(g_sum[i], reg_l1) / (h_sum[i] + reg_l2); }
//     }
// }

//=========================================================================
//                                                                         
//   ####   ####   #####   #####    #####                                
//  ##     ##     ##   ##  ##  ##   ##                                   
//   ###   ##     ##   ##  #####    #####                                
//     ##  ##     ##   ##  ##  ##   ##                                   
//  ####    ####   #####   ##   ##  #####                                
//                                                                         
//=========================================================================

inline static double CalScoreNoL1(
    const double g_sum,
    const double h_sum,
    const double reg_l1, // unused
    const double reg_l2
) {
    return Sqr(g_sum) / (h_sum + reg_l2);
};

inline static double CalScoreL1(
    const double g_sum, 
    const double h_sum, 
    const double reg_l1, 
    const double reg_l2
) {
    return Sqr(ThresholdL1(g_sum, reg_l1)) / (h_sum + reg_l2);
};

inline static void Softmax(std::vector<double>& rec) {
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
        double erec = exp(rec[i] - wmax);
        rec[i] = erec;
        wsum += erec;
    }
    double wsumi = 1/wsum;
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] *= wsumi;
    }
}

inline static double Log_sum_exp(const std::vector<double>& rec) {
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
        wsum += exp(rec[i] - wmax);
    }
    return log(wsum) + wmax;
}

#endif /* GBDTMO_MATH_H */
