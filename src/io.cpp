#include "io.h"
#include "string_utils.h"

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#define PRECISION 16

//=======================================================================
//                                                                       
//  ####    ##   ##  ###    ###  #####                                 
//  ##  ##  ##   ##  ## #  # ##  ##  ##                                
//  ##  ##  ##   ##  ##  ##  ##  #####                                 
//  ##  ##  ##   ##  ##      ##  ##                                    
//  ####     #####   ##      ##  ##                                    
//                                                                       
//=======================================================================

void DumpTrees(std::vector<Tree>& trees, const char* path) {
    std::ofstream outfile;
    outfile.open(path);
    size_t t = 0;
    for (auto& tree : trees) {
        outfile << "Booster[" << t << "]:\n";
        for (auto it : tree.nonleaf) {
            auto v = it.second;
            outfile << "\t" << it.first << "," << v.parent << "," << v.left << "," << v.right << ","
                    << v.column << ",";
            outfile << std::scientific << std::setprecision(PRECISION) << v.threshold << std::endl;
        }

        // std::cout << "tree.leaf.size() = " << tree.leaf.size() << std::endl;
        for (auto& it : tree.leaf) {
            auto v = it.second;
            outfile << "\t\t" << it.first << ",";
            for (size_t i = 0; i < v.values.size(); ++i) {
                outfile << std::scientific << std::setprecision(PRECISION) << v.values[i];
                if (i < v.values.size() - 1) {
                    outfile << ",";
                } else {
                    outfile << std::endl;
                }
            }
        }
        ++t;
    }
    outfile.close();
}

// void DumpState(std::vector<Tree>& trees) {
    
//     for (size_t t = 0; t < trees.size(); ++t) {
//         auto tree = trees[t];
//         // outfile << "Booster[" << t << "]:\n";
//         for (auto it : tree.nonleaf) {
//             auto v = it.second;
//             outfile << "\t" << it.first << "," << v.parent << "," << v.left << "," << v.right << "," << v.column << ",";
//             outfile << std::scientific << std::setprecision(PRECISION) << v.threshold << std::endl;
//         }

//         for (auto& it : tree.leaf) {
//             auto v = it.second;
//             outfile << "\t\t" << it.first << ",";
//             for (size_t i = 0; i < v.values.size(); ++i) {
//                 outfile << std::scientific << std::setprecision(PRECISION) << v.values[i];
//                 if (i < v.values.size() - 1) {
//                     outfile << ",";
//                 } else {
//                     outfile << std::endl;
//                 }
//             }
//         }
//     }
//     // outfile.close();
// }

//====================================================================
//                                                                    
//  ##       #####     ###    ####                                  
//  ##      ##   ##   ## ##   ##  ##                                
//  ##      ##   ##  ##   ##  ##  ##                                
//  ##      ##   ##  #######  ##  ##                                
//  ######   #####   ##   ##  ####                                  
//                                                                    
//====================================================================

void LoadTrees(std::vector<Tree>& trees, const char* path) {
    std::ifstream infile(path);
    std::string line;
    std::vector<std::string> contents;
    Tree tree_(false);
    int t = 0, num;
    while (getline(infile, line)) {
        //Booster
        if (line.find("Booster") == 0) {
            if (t > 0) {
                trees.push_back(tree_);
                tree_.clear();
            }
            ++t;
        } else {
            contents.resize(0);
            line = lstrip(line, "\t");
            split(line, contents, ",");
            num = std::stoi(contents[0]);
            if (num < 0) {
                //nonleaf
                NonLeafNode node;
                node.parent = std::stoi(contents[1]);
                node.left = std::stoi(contents[2]);
                node.right = std::stoi(contents[3]);
                node.column = std::stoi(contents[4]);
                node.threshold = std::stod(contents[5]);
                tree_.nonleaf.emplace(num, node);
            } else {
                //leaf
                LeafNode node;
                node.values.resize(contents.size() - 1);
                for (int i = 1; i < contents.size(); ++i) { node.values[i - 1] = std::stod(contents[i]); }
                tree_.leaf.emplace(num, node);
            }
        }
    }
    infile.close();
}