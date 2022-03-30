#ifndef GBDTMO_MATH_H
#define GBDTMO_MATH_H

#include <vector>
#include <math.h>
#include <unordered_set>

// __attribute__ ((pure)
#define GBDTMO_MATH_FUNC [[gnu::pure]] inline static

template<typename T>
GBDTMO_MATH_FUNC
T Sum(std::vector<T> x) {
    T s = 0;
    for (auto& i : x) { s += i; }
    return s;
}

template<typename T>
GBDTMO_MATH_FUNC
T Sqr(T x) { return x * x; }

inline static float ThresholdL1(float w, float reg_l1) {
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

GBDTMO_MATH_FUNC
float CalWeightNoL1(
    const float g_sum,
    const float h_sum,
    const float reg_l1,
    const float reg_l2
) {
    return -g_sum / (h_sum + reg_l2);
}

GBDTMO_MATH_FUNC
float CalWeightL1(
    const float g_sum,
    const float h_sum,
    const float reg_l1,
    const float reg_l2
) {
    return -ThresholdL1(g_sum, reg_l1) / (h_sum + reg_l2);
}

//=========================================================================
//                                                                         
//   ####   ####   #####   #####    #####                                
//  ##     ##     ##   ##  ##  ##   ##                                   
//   ###   ##     ##   ##  #####    #####                                
//     ##  ##     ##   ##  ##  ##   ##                                   
//  ####    ####   #####   ##   ##  #####                                
//                                                                         
//=========================================================================

GBDTMO_MATH_FUNC
float CalScoreNoL1(
    const float g_sum,
    const float h_sum,
    const float reg_l1, // unused
    const float reg_l2
) {
    return Sqr(g_sum) / (h_sum + reg_l2);
};

GBDTMO_MATH_FUNC
float CalScoreL1(
    const float g_sum, 
    const float h_sum, 
    const float reg_l1, 
    const float reg_l2
) {
    return Sqr(ThresholdL1(g_sum, reg_l1)) / (h_sum + reg_l2);
};

//================================================================================================
//                                                                                                
//   ####   #####   #####  ######  ###    ###    ###    ##    ##                                
//  ##     ##   ##  ##       ##    ## #  # ##   ## ##    ##  ##                                 
//   ###   ##   ##  #####    ##    ##  ##  ##  ##   ##    ####                                  
//     ##  ##   ##  ##       ##    ##      ##  #######   ##  ##                                 
//  ####    #####   ##       ##    ##      ##  ##   ##  ##    ##                                
//                                                                                                
//================================================================================================

GBDTMO_MATH_FUNC
void Softmax(std::vector<float>& rec) {
    float wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    float wsum = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
        float erec = exp(rec[i] - wmax);
        rec[i] = erec;
        wsum += erec;
    }
    float wsumi = 1/wsum;
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] *= wsumi;
    }
}

GBDTMO_MATH_FUNC
float Log_sum_exp(const std::vector<float>& rec) {
    float wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    float wsum = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
        wsum += exp(rec[i] - wmax);
    }
    return log(wsum) + wmax;
}

//==============================================================================
//                                                                              
//   ####  ##        ###    ###    ###  #####                                 
//  ##     ##       ## ##   ## #  # ##  ##  ##                                
//  ##     ##      ##   ##  ##  ##  ##  #####                                 
//  ##     ##      #######  ##      ##  ##                                    
//   ####  ######  ##   ##  ##      ##  ##                                    
//                                                                              
//==============================================================================


// constexpr const T& clamp( const T& v, const T& lo, const T& hi, Compare comp )
// {
//     return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
// }

// template<class T>
// T clamp(const T& v, const T& lo, const T& hi) {
//     return std::max(lo, std::min(v, hi));
// }


#endif /* GBDTMO_MATH_H */
