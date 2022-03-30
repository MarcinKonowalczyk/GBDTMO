#ifndef GBDTMO_LOSS_H
#define GBDTMO_LOSS_H

#include <iostream>
#include <exception>
#include <stdexcept>
#include <cstring>
#include "datastruct.h"
#include "mathFunc.h"

//TODO: make it easier to implement new loss objective

// grad function: dataset, g, h, rows, columns
// score function: dataset, rows, columns
// function with "column": only used for multiple predictions one by one

typedef void   (*func_grad)           (const Dataset& data, const size_t out_dim, float* const g, float* const h);
typedef float (*func_score)          (const Dataset& data, const size_t out_dim, const float* const g);
typedef float (*func_partial_score)  (const Dataset& data, const size_t out_dim, const float* const g, const bool score_train);
typedef float (*func_metric)         (const Dataset& data, const size_t out_dim, const float* const g, const bool score_train);

void   mse_grad              (const Dataset&, const size_t, float* const, float* const);
float mse_score             (const Dataset&, const size_t, const float* const);
float mse_partial_score     (const Dataset&, const size_t, const float* const, const bool);
void   bce_grad              (const Dataset&, const size_t, float* const, float* const);
float bce_score             (const Dataset&, const size_t, const float* const);
float bce_partial_score     (const Dataset&, const size_t, const float* const, const bool);
void   ce_grad               (const Dataset&, const size_t, float* const, float* const);
float ce_score              (const Dataset&, const size_t, const float* const);
void   ce_grad_column        (const Dataset&, const size_t, float* const, float* const);
float ce_score_column       (const Dataset&, const size_t, const float* const);
float acc_binary            (const Dataset&, const size_t, const float* const, const bool);
float acc_multiclass        (const Dataset&, const size_t, const float* const);
float acc_multiclass_column (const Dataset&, const size_t, const float* const);

struct Objective {
    bool constHessian = true;
    float hessian = 1.0;
    float largerBetter = false;
    func_grad f_grad;
    func_score f_score;
    func_partial_score f_partial_score;
    func_metric f_metric;

    Objective() {};

    Objective(const Loss name) {
        switch (name) {
            case mse : {
                f_grad = mse_grad;
                f_score = mse_score;
                f_partial_score = mse_partial_score;
                f_metric = mse_partial_score;
            } break;
            case bce : {
                constHessian = false;
                largerBetter = true;
                f_grad = bce_grad;
                f_score = bce_score;
                f_partial_score = bce_partial_score;
                f_metric = acc_binary;
            } break;
            case ce : {
                std::cout << "Not implemented yet!\n";
                throw;
                constHessian = false;
                largerBetter = true;
                f_grad = ce_grad;
                f_score = ce_score;
                f_partial_score = nullptr;
                f_metric = nullptr;
                // f_metric = acc_multiclass;
            } break;
            case ce_column : {
                std::cout << "Not implemented yet!\n";
                throw;
                constHessian = false;
                largerBetter = true;
                f_grad = ce_grad_column;
                f_score = ce_score_column;
                f_partial_score = nullptr;
                f_metric = nullptr;
                // f_metric = acc_multiclass_column;
            } break;
        }
    }
};

#endif /* GBDTMO_LOSS_H */
