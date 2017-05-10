# cudaFilter
demonstrates a tricky way (but still not very optimized) to execute a recursive image filter with CUDA.

In this algorithm result is taken through 1-D consecutive filtering of rows and columns. It is known that we can avoid the non-coalesced memory access by transposing the rows and make each CUDA thread process its row as a column (first part).

It is assumed that bf[], bb[] and B coefficients are prepared in the main() 
