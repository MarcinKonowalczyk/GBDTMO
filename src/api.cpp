#include "booster.h"

extern "C" {

//=======================================================================================
//                                                                                       
//   ####  #####  ######  ######  #####  #####     ####                                
//  ##     ##       ##      ##    ##     ##  ##   ##                                   
//   ###   #####    ##      ##    #####  #####     ###                                 
//     ##  ##       ##      ##    ##     ##  ##      ##                                
//  ####   #####    ##      ##    #####  ##   ##  ####                                 
//                                                                                       
//=======================================================================================

void SetBin(BoosterBase* foo, uint16_t* bins, double* values) { foo->set_bin(bins, values); }
void SetGH(BoosterBase* foo, double* x, double* y) { foo->set_gh(x, y); }
void SetTrainData(BoosterBase* foo, uint16_t* maps, double* features, double* preds, int n) { foo->set_train_data(maps, features, preds, n); }
void SetEvalData(BoosterBase* foo, uint16_t* maps, double* features, double* preds, int n) { foo->set_eval_data(maps, features, preds, n); }
void SetTrainLabelDouble(BoosterBase* foo, double* label) { foo->set_train_label(label); }
void SetTrainLabelInt(BoosterBase* foo, int32_t* label) { foo->set_train_label(label); }
void SetEvalLabelDouble(BoosterBase* foo, double* label) { foo->set_eval_label(label); }
void SetEvalLabelInt(BoosterBase* foo, int32_t* label) { foo->set_eval_label(label); }

//========================================================================
//                                                                        
//  ##      ##   #####   #####    ##  ##                                
//  ##      ##  ##   ##  ##  ##   ## ##                                 
//  ##  ##  ##  ##   ##  #####    ####                                  
//  ##  ##  ##  ##   ##  ##  ##   ## ##                                 
//   ###  ###    #####   ##   ##  ##  ##                                
//                                                                        
//========================================================================


void Boost(BoosterBase* foo) { foo->growth(); foo->update(); }
void Train(BoosterBase* foo, int num_rounds) { foo->train(num_rounds); }
void Predict(BoosterBase* foo, double* features, double* preds, int n, int num_trees) { foo->predict(features, preds, n, num_trees); }
void Reset(BoosterBase* foo) { foo->reset(); }

//===============================================
//                                               
//  ##   #####                                 
//  ##  ##   ##                                
//  ##  ##   ##                                
//  ##  ##   ##                                
//  ##   #####                                 
//                                               
//===============================================

void Dump(BoosterBase* foo, const char* path) { foo->dump(path); }
void Load(BoosterBase* foo, const char* path) { foo->load(path); }

//=============================================================
//                                                             
//  ##  ##     ##  ##  ######                                
//  ##  ####   ##  ##    ##                                  
//  ##  ##  ## ##  ##    ##                                  
//  ##  ##    ###  ##    ##                                  
//  ##  ##     ##  ##    ##                                  
//                                                             
//=============================================================

// Define the default hyperparameters in the shared object itself, not on the python side
HyperParameters DefaultHyperParameters() {
    return (HyperParameters) {
        1, // inp_dim
        1, // out_dim
        "mse", // loss
        5, // max_depth
        32, // max_leaves
        0, // seed
        5, // min_samples
        0.2, // lr
        0.0, // reg_l1
        1.0, // reg_l2
        1e-3, // gamma
        0.0f, // base_score
        0, // early_stop
        true, // verbose
        16, // max_caches
        0, // topk
        true, // one_side
    };
}

BoosterMulti* MultiNew(HyperParameters hp) { return new BoosterMulti(hp); }
BoosterSingle* SingleNew(HyperParameters hp) { return new BoosterSingle(hp); }

}