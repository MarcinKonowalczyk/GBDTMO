#include "string_utils.h"

std::string _do_strip(
    const std::string &str,
    const int striptype,
    const std::string &chars
) {
    size_t len = str.size(), i, j, charslen = chars.size();
    if (charslen == 0) {
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < len && ::isspace(str[i])) ++i;
        }

        j = len;
        if (striptype != LEFTSTRIP) {
            do --j; while (j >= i && ::isspace(str[j]));
            ++j;
        }
    } else {
        const char *sep = chars.c_str();
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < len && memchr(sep, str[i], charslen)) ++i;
        }

        j = len;
        if (striptype != LEFTSTRIP) {
            do --j; while (j >= i && memchr(sep, str[j], charslen));
            ++j;
        }
    }

    if (i == 0 && j == len) {
        return str;
    } else {
        return str.substr(i, j - i);
    }
}

void split_whitespace(
    const std::string &str,
    std::vector<std::string> &result,
    int maxsplit
) {
    std::string::size_type i, j, len = str.size();
    for (i = j = 0; i < len;) {
        while (i < len && ::isspace(str[i])) ++i;
        j = i;

        while (i < len && !::isspace(str[i])) ++i;

        if (j < i) {
            if (maxsplit-- <= 0) break;
            result.push_back(str.substr(j, i - j));
            while (i < len && ::isspace(str[i])) ++i;
            j = i;
        }
    }
    if (j < len) {
        result.push_back(str.substr(j, len - j));
    }
}

void split(
    const std::string &str,
    std::vector<std::string> &result,
    const std::string &sep,
    int maxsplit
) {
    result.clear();
    if (maxsplit < 0) maxsplit = INT_MAX;//result.max_size();
    if (sep.size() == 0) {
        split_whitespace(str, result, maxsplit);
        return;
    }

    std::string::size_type i, j, len = str.size(), n = sep.size();
    i = j = 0;
    while (i + n <= len) {
        if (str[i] == sep[0] && str.substr(i, n) == sep) {
            if (maxsplit-- <= 0) break;
            result.push_back(str.substr(j, i - j));
            i = j = i + n;
        } else {
            ++i;
        }
    }
    result.push_back(str.substr(j, len - j));
}

std::string zfill(const std::string &str, const int width) {
    size_t len = str.size();
    if (len >= width) return str;
    std::string s(str);
    size_t fill = width - len;
    s = std::string(fill, '0') + s;

    if (s[fill] == '+' || s[fill] == '-') {
        s[0] = s[fill];
        s[fill] = '0';
    }

    return s;
}