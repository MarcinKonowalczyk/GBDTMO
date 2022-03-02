//
// Created by zhang on 19-7-4.
//

#ifndef GBDTMO_IO_H
#define GBDTMO_IO_H

#include "tree.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits.h>

// functions for string processing
std::string lstrip(const std::string &str, const std::string &chars = "");

std::string rstrip(const std::string &str, const std::string &chars = "");

std::string strip(const std::string &str, const std::string &chars = "");

void split(const std::string &str, std::vector<std::string> &result, const std::string &sep = "", int maxsplit = -1);

std::string zfill(const std::string &str, int width);

//functions for dump and load the learned trees (to and from txt file)
void DumpTrees(std::vector<Tree> &, const char *);

void LoadTrees(std::vector<Tree> &, const char *);

#endif /* GBDTMO_IO_H */
