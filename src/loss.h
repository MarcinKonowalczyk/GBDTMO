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

typedef void   (*func_grad)   (const Dataset& data, const size_t out_dim, double* g, double* h);
typedef double (*func_score)  (const Dataset& data, const size_t out_dim);
typedef double (*func_metric) (const Dataset& data, const size_t out_dim);

void   mse_grad              (const Dataset&, const size_t, double*, double*);
double mse_score             (const Dataset&, const size_t);
void   bce_grad              (const Dataset&, const size_t, double*, double*);
double bce_score             (const Dataset&, const size_t);
void   ce_grad               (const Dataset&, const size_t, double*, double*);
double ce_score              (const Dataset&, const size_t);
void   ce_grad_column        (const Dataset&, const size_t, double*, double*);
double ce_score_column       (const Dataset&, const size_t);
double acc_binary            (const Dataset&, const size_t);
double acc_multiclass        (const Dataset&, const size_t);
double acc_multiclass_column (const Dataset&, const size_t);

struct Objective {
    bool constHessian = true;
    double hessian = 1.0;
    double largerBetter = false;
    func_grad f_grad;
    func_score f_score;
    func_metric f_metric;

    Objective() {};

    Objective(const Loss name) {
        switch (name) {
            case mse : {
                f_grad = mse_grad;
                f_score = mse_score;
                f_metric = mse_score;
            } break;
            case ce : {
                constHessian = false;
                largerBetter = true;
                f_grad = ce_grad;
                f_score = ce_score;
                f_metric = acc_multiclass;
            } break;
            case ce_column : {
                constHessian = false;
                largerBetter = true;
                f_grad = ce_grad_column;
                f_score = ce_score_column;
                f_metric = acc_multiclass_column;
            } break;
            case bce : {
                constHessian = false;
                largerBetter = true;
                f_grad = bce_grad;
                f_score = bce_score;
                f_metric = acc_binary;
            } break;
        }
    }
};

#endif /* GBDTMO_LOSS_H */
