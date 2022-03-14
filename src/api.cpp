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

// void SetGH(BoosterBase* foo, double* x, double* y) { foo->set_gh(x, y); }
void SetTrainData(BoosterBase* foo, double* features, double* preds, int n) { foo->set_train_data(features, preds, n); }
void SetEvalData(BoosterBase* foo, double* features, double* preds, int n) { foo->set_eval_data(features, preds, n); }
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

void CalcTrainMaps(BoosterBase* foo) { foo->calc_train_maps(); }
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

size_t GetNTrees(BoosterBase* foo) { return foo->trees.size(); }
void GetNonleafSizes(BoosterBase* foo, uint16_t* nonleaf_sizes) { foo->dump_nonleaf_sizes(nonleaf_sizes); }
void GetLeafSizes(BoosterBase* foo, uint16_t* leaf_sizes) { foo->dump_leaf_sizes(leaf_sizes); }
void GetNonleafNodes(BoosterBase* foo, int* trees, double* thresholds) { foo->dump_nonleaf_nodes(trees, thresholds); }
void GetLeafNodes(BoosterBase* foo, double* leaves) { foo->dump_leaf_nodes(leaves); }

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
        32, // max_bins
    };
}

BoosterMulti* MultiNew(HyperParameters hp) { return new(std::nothrow) BoosterMulti(hp); }
BoosterSingle* SingleNew(HyperParameters hp) { return new(std::nothrow) BoosterSingle(hp); }
void Delete(BoosterBase* foo) { delete foo; }
}