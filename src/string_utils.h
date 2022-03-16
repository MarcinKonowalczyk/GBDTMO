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
std::string _do_strip(const std::string& str, const int striptype, const std::string& chars);

inline static
std::string strip(const std::string& str, const std::string& chars = "") {
    return _do_strip(str, BOTHSTRIP, chars);
}

inline static
std::string lstrip(const std::string& str, const std::string& chars = "") {
    return _do_strip(str, LEFTSTRIP, chars);
}

inline static
std::string rstrip(const std::string& str, const std::string& chars = "") {
    return _do_strip(str, RIGHTSTRIP, chars);
}

void split_whitespace(
    const std::string& str,
    std::vector<std::string>& result,
    int maxsplit);

void split(
    const std::string& str,
    std::vector<std::string>& result,
    const std::string& sep = "",
    int maxsplit = -1
);

std::string zfill(const std::string& str, const int width);


//========================================================================================
//                                                                                        
//  #####   #####   #####    ###    ###    ###    ######                                
//  ##     ##   ##  ##  ##   ## #  # ##   ## ##     ##                                  
//  #####  ##   ##  #####    ##  ##  ##  ##   ##    ##                                  
//  ##     ##   ##  ##  ##   ##      ##  #######    ##                                  
//  ##      #####   ##   ##  ##      ##  ##   ##    ##                                  
//                                                                                        
//========================================================================================

// https://stackoverflow.com/a/26221725/2531987

// #include <memory>
// #include <string>
// #include <stdexcept>

// template<typename ... Args>
// std::string string_format( const std::string& format, Args ... args )
// {
//     int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
//     if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
//     auto size = static_cast<size_t>( size_s );
//     std::unique_ptr<char[]> buf( new char[ size ] );
//     std::snprintf( buf.get(), size, format.c_str(), args ... );
//     return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
// }

#endif /* GBDTMO_STRING_UTILS_H */