#include "io.h"
#include "string_utils.h"

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#define PRECISION 6

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
        for (auto& it : tree.nonleafs) {
            auto v = it.second;
            outfile << "\t" << it.first << "," << v.parent << "," << v.left << "," << v.right << ","
                    << v.column << ",";
            outfile << std::scientific << std::setprecision(PRECISION) << v.threshold << std::endl;
        }

        for (auto& it : tree.leafs) {
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
    Tree _tree(false);
    int t = 0, num;
    while (getline(infile, line)) {
        //Booster
        if (line.find("Booster") == 0) {
            if (t > 0) {
                trees.push_back(_tree);
                _tree.clear();
            }
            ++t;
        } else {
            contents.resize(0);
            line = lstrip(line, "\t");
            split(line, contents, ",");
            num = std::stoi(contents[0]);
            if (num < 0) { // nonleaf
                int parent = std::stoi(contents[1]);
                int column = std::stoi(contents[4]);
                float threshold = std::stod(contents[5]);
                auto node = NonLeafNode(parent, column, -1, threshold);
                node.left = std::stoi(contents[2]);
                node.right = std::stoi(contents[3]);
                _tree.nonleafs.emplace(num, node);
            } else { // leaf
                auto values = std::vector<std::pair<float, int>>();
                for (size_t i = 1; i < contents.size(); ++i) {
                    values.push_back(std::pair<float, int>(std::stod(contents[i]), i - 1));
                }
                auto node = LeafNode(contents.size() - 1, values);
                _tree.leafs.emplace(num, node);
            }
        }
    }
    infile.close();
}