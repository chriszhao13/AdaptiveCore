

#ifndef CUTS_GPU_MEMORY_ALLOCATION_H
#define CUTS_GPU_MEMORY_ALLOCATION_H
#include "./graph.h"


void malloc_graph_gpu_memory(Graph &g,G_pointers &p);
void get_results_from_gpu(Graph &g,G_pointers &p);
void free_graph_gpu_memory(G_pointers &p);


#define chkerr(ans) { chkerr1((ans), __FILE__, __LINE__); }
inline void chkerr1(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif //CUTS_GPU_MEMORY_ALLOCATION_H




