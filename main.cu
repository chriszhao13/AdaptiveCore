#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "./include/gpu_memory_allocation.h"
#include "./src/AdaptiveCore.cc"

int rep = 1;

int cmp(const void *a, const void *b)
{
    return *(unsigned int *)a - *(unsigned int *)b;
}

void PrintMax(uint *result, uint n)
{

    uint max = 0;
    uint min = 10;
    uint id = 0;
    for (int i = 0; i < n; i++)
    {

        if (max < result[i])
        {
            max = result[i];
            id = i;
        }
        if (min > result[i])
            min = result[i];
    }

    cout << " Maxdegree = " << max << endl;
    cout << " Id = " << id << endl;

    // cout << " Min K = " << min << endl;
}

template <class T>
void repSimulation(int (*kern)(T), Graph &g)
{
    float sum = 0;

    // int rep = 10; // number of iterations...
    uint max_time = 0;

    for (int i = 0; i < rep; i++)
    {

        // cout <<"  !!!  " << i << "ms "<<endl;
        int t = (*kern)(g);
        // cout <<"??1??" << t<< endl;
        // cudaDeviceSynchronize();
        // cout <<"??2??" << t << endl;

        cout << t << "ms ";

        if (max_time < t)
            max_time = t;
        sum += t;
    }

    sum = sum - max_time;

    cout << RED << endl
         << "Ave: " << sum * 1.0 / (rep - 1) << " ms" << RESET << endl;
}

void STDdegrees(Graph &g)
{
    double sum = std::accumulate(g.degrees, g.degrees + g.V, 0.0);
    double mean = sum / g.V;

    std::vector<double> diff(g.V);
    std::transform(g.degrees, g.degrees + g.V, diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / g.V);
#ifdef HELLO
    cout << "STD: " << stdev;
#endif
}

int main(int argc, char *argv[])
{

    if (argc < 4)
    {
        cout << "./kcore file deviceID Repeattime" << endl;
        exit(-1);
    }

    std::string ds = argv[1];

    cudaSetDevice(std::atoi(argv[2]));

    rep = std::atoi(argv[3]);
    cudaFree(0);

    cout << "Graph loading Started... " << endl;

    Graph g(ds);
    cout << endl << "Grapg Name: " << ds << "V: " << g.V << " undirected E: " << g.E << endl;
    cout << "AdaptiveCore: ";
    repSimulation(AdaptiveCore, g);
    cout << "Kmax: " << g.kmax << endl;

    return 0;
}
