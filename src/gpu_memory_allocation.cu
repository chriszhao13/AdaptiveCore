
#include "../include/gpu_memory_allocation.h"

void malloc_graph_gpu_memory(Graph &g, G_pointers &p)
{
    chkerr(cudaMalloc(&(p.neighbors), g.neighbors_offset[g.V] * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors, g.neighbors, g.neighbors_offset[g.V] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.neighbors_offset), (g.V + 1) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.neighbors_offset, g.neighbors_offset, (g.V + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    chkerr(cudaMalloc(&(p.degrees), (g.V) * sizeof(unsigned int)));
    chkerr(cudaMemcpy(p.degrees, g.degrees, (g.V) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    // std::cout<<"memory graph p = "<<p.neighbors[0]<<endl;
}

void free_graph_gpu_memory(G_pointers &p)
{
    chkerr(cudaFree(p.neighbors));
    chkerr(cudaFree(p.neighbors_offset));
    chkerr(cudaFree(p.degrees));
}
