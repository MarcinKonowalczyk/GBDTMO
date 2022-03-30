#include "histogram.h"
#include <algorithm>
#include <utility>

//=====================================================================================================================================
//                                                                                                                                     
//  ##   ##  ##     ##  ##   #####   ##   ##  #####         ####   #####   ##   ##  ##     ##  ######                                
//  ##   ##  ####   ##  ##  ##   ##  ##   ##  ##           ##     ##   ##  ##   ##  ####   ##    ##                                  
//  ##   ##  ##  ## ##  ##  ##   ##  ##   ##  #####        ##     ##   ##  ##   ##  ##  ## ##    ##                                  
//  ##   ##  ##    ###  ##   #####   ##   ##  ##           ##     ##   ##  ##   ##  ##    ###    ##                                  
//   #####   ##     ##  ##  ##        #####   #####         ####   #####    #####   ##     ##    ##                                  
//                                                                                                                                     
//=====================================================================================================================================

// Modification of std::unique implementaiton form:
// https://en.cppreference.com/w/cpp/algorithm/unique
// It also counts the number of elements in each 
template<typename T>
void inplace_unique_with_count(
    std::vector<T>& vector,
    std::vector<size_t>& counts
) {
    // Special cases
    if (vector.size() == 0) {
        return;
    } else if (vector.size() == 1) {
        counts.push_back(1);
        return;
    }

    // Iterators through the vector
    auto first = vector.begin(), last = vector.end();
    auto result = first;

    std::sort(first, last); // Sort the vector; NOTE: std::sort does not invalidate the iterators

    // Keep track of how many elemnts to skip in the count as the pointers get away from one another
    int count, skip = 0;
    do {
        if (!(*result == *first)) {
            if (++result != first) *result = std::move(*first);
            count = first - result - skip;
            counts.push_back(count + 1);
            skip += count;
        }
    } while (++first != last);

    counts.push_back(first - result - skip); // Count the last element
    vector.erase(++result, last); // Trim vector to length
}

//=======================================================================================
//                                                                                       
//  #####   ##  ##     ##         ####   #####   ##                                    
//  ##  ##  ##  ####   ##        ##     ##   ##  ##                                    
//  #####   ##  ##  ## ##        ##     ##   ##  ##                                    
//  ##  ##  ##  ##    ###        ##     ##   ##  ##                                    
//  #####   ##  ##     ##         ####   #####   ######                                
//                                                                                       
//=======================================================================================

// Construt bin edges based on the column of the features matrix
void construct_bin_column(
    std::vector<float> features_column, // Note: don't pass by reference. Copy. 'feature' is not modified outside of the function.
    std::vector<float>& bins_column,
    const uint16_t max_bins
) {
    size_t n = features_column.size();
    // Inplace find unique elements and their counts
    std::vector<size_t> counts;
    inplace_unique_with_count(features_column, counts);

    size_t N = features_column.size();
    if (N <= 1) {
        // pass
    } else if (N == 2) {
        float split = features_column[0] * counts[0] + features_column[1] * counts[1];
        bins_column.push_back(split);
    } else if (N <= max_bins) {
        bins_column.resize(N-1);
        for(auto it = features_column.begin(); it < features_column.end()-1; ++it) {
            bins_column.push_back(*it*0.5 + *(it+1)*0.5);
        }
    } else { // N > max_bins
        size_t bin_index = 0;
        float p = N / max_bins;
        size_t ccount = 0;
        bins_column.resize(max_bins);
        for(size_t i = 0; i < N; ++i) {
            ccount += counts[i]; // Cumulative count
            if (ccount >= p) {
                bins_column[bin_index++] = features_column[i];
                p = ccount + (n - ccount) / float(max_bins - bin_index);
            }
            if (bin_index == max_bins - 1) break; // Stop at the penultimate bin
        }
        bins_column[bin_index++] = features_column[N-1]; // Assign last bin to last feature
    }
};

//=============================================================================================
//                                                                                             
//  ###    ###    ###    #####          ####   #####   ##                                    
//  ## #  # ##   ## ##   ##  ##        ##     ##   ##  ##                                    
//  ##  ##  ##  ##   ##  #####         ##     ##   ##  ##                                    
//  ##      ##  #######  ##            ##     ##   ##  ##                                    
//  ##      ##  ##   ##  ##             ####   #####   ######                                
//                                                                                             
//=============================================================================================

// Calculate binning of a single column of the features matrix
void map_bin_column(
    const std::vector<float> features_column,
    std::vector<uint16_t>& map_column,
    std::vector<float>& bins
) {
    for (size_t i = 0; i < features_column.size(); ++i) {
        auto it = std::lower_bound(bins.begin(), bins.end(), features_column[i]);
        map_column[i] = static_cast<uint16_t>(it - bins.begin());
    }
};

// // Calculate binning of each colum of the features matrix, and the corresponding histogram map
// void calculate_histogram_map_column(
//     const std::vector<float> features_column,
//     std::vector<uint16_t>& map_column,
//     std::vector<float>& bins_column,
//     const size_t n,
//     const size_t inp_dim,
//     const uint16_t max_bins
// ) {
//     // Construct bins for the column
//     construct_bin_column(features_column, bins_column, max_bins);

//     // Map features in the particular column
//     // NOTE: This means that maps are column-major
//     map_bin_column(features_column, map_column, bins_column);
// };
