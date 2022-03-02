#ifndef GBDTMO_LOSS_H
#define GBDTMO_LOSS_H

#include <iostream>
#include <exception>
#include <stdexcept>
#include <cstring>
#include "dataStruct.h"
#include "mathFunc.h"

//TODO: make it easier to implement new loss objective

// grad function: dataset, g, h, rows, columns
// score function: dataset, rows, columns
// function with "column": only used for multiple predictions one by one

typedef void (*func_grad) (const Dataset& data, const int n, const int out_dim, double* g, double* h);
typedef double (*func_score) (const Dataset& data, const int n, const int out_dim);
typedef double (*func_metric) (const Dataset& data, const int n, const int out_dim);

void   mse_grad              (const Dataset&, const int, const int, double*, double*);
double mse_score             (const Dataset&, const int, const int);
void   bce_grad              (const Dataset&, const int, const int, double*, double*);
double bce_score             (const Dataset&, const int, const int);
void   ce_grad               (const Dataset&, const int, const int, double*, double*);
double ce_score              (const Dataset&, const int, const int);
void   ce_grad_column        (const Dataset&, const int, const int, double*, double*);
double ce_score_column       (const Dataset&, const int, const int);
double acc_binary            (const Dataset&, const int, const int);
double acc_multiclass        (const Dataset&, const int, const int);
double acc_multiclass_column (const Dataset&, const int, const int);

struct Objective {
    bool constHessian = true;
    double hessian = 1.0f;
    double largerBetter = false;
    func_grad f_grad;
    func_score f_score;
    func_metric f_metric;

    Objective() {};

    Objective(const char* name) {
        if (strcmp(name, "mse") == 0) {
            f_grad = mse_grad;
            f_score = mse_score;
            f_metric = mse_score;
        } else if (strcmp(name, "ce") == 0) {
            constHessian = false;
            largerBetter = true;
            f_grad = ce_grad;
            f_score = ce_score;
            f_metric = acc_multiclass;
        } else if (strcmp(name, "ce_column") == 0) {
            constHessian = false;
            largerBetter = true;
            f_grad = ce_grad_column;
            f_score = ce_score_column;
            f_metric = acc_multiclass_column;
        } else if (strcmp(name, "bce") == 0) {
            constHessian = false;
            largerBetter = true;
            f_grad = bce_grad;
            f_score = bce_score;
            f_metric = acc_binary;
        } else {
            std::string s = "Objective type name must be in [mse, bce, ce, ce_column]";
            std::cout << s << std::endl;
            throw std::runtime_error(s);
        }
    }
};

#endif /* GBDTMO_LOSS_H */
