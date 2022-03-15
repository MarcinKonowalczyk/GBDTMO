#ifndef GBDTMO_STRING_UTILS_H
#define GBDTMO_STRING_UTILS_H

#include <string>
#include <vector>

// #include <algorithm>
// #include <cctype>
// #include <cstring>
// #include <sstream>

#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2

extern
std::string _do_strip(const std::string &str, int striptype, const std::string &chars);

inline static
std::string strip(const std::string &str, const std::string &chars = "") {
    return _do_strip(str, BOTHSTRIP, chars);
}

inline static
std::string lstrip(const std::string &str, const std::string &chars = "") {
    return _do_strip(str, LEFTSTRIP, chars);
}

inline static
std::string rstrip(const std::string &str, const std::string &chars = "") {
    return _do_strip(str, RIGHTSTRIP, chars);
}

void split_whitespace(const std::string &str, std::vector<std::string> &result, int maxsplit);

void split(
    const std::string &str,
    std::vector<std::string> &result,
    const std::string &sep = "",
    int maxsplit = -1
);

std::string zfill(const std::string &str, int width);

#endif /* GBDTMO_STRING_UTILS_H */