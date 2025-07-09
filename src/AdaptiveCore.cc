#define INTRA_1_CORE_warpdyn
#define SPlit_1_Vertex
#define inter_1_warp
#define JUMPCORE
#define GLOBAL_2_QUEUE
#define break_num1 1024

// #define PRINTXXX

__device__ inline void writeToBuffer_loop(unsigned int *glBuffer, unsigned int loc, unsigned int v)
{
    assert(loc < GLBUFFER_SIZE);
    glBuffer[loc] = v;
}

__device__ inline unsigned int readFrom_SM_Blk_Queue(unsigned int *shBuffer, unsigned int *glBuffer, unsigned int loc)
{
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    unsigned int v;
    if (loc < MAX_NV)
        v = shBuffer[loc];
    else
        v = glBuffer[loc - MAX_NV];
    return v;
}

__device__ inline void writeTo_SM_Blk_Queue_scan(unsigned int *shBuffer, unsigned int *glBuffer, unsigned int loc, unsigned int v)
{
    assert(loc < GLBUFFER_SIZE + MAX_NV);

    if (loc < MAX_NV)
    {
        shBuffer[loc] = v;
        return;
    }
    glBuffer[loc - MAX_NV] = v;
}

__device__ inline void writeTo_SM_Blk_Queue_loop(unsigned int *shBuffer, unsigned int *glBuffer, unsigned int loc, unsigned int v)
{
    assert(loc < GLBUFFER_SIZE + MAX_NV);
    if (loc < MAX_NV)
    {
        shBuffer[loc] = v;
        return;
    }
    glBuffer[loc - MAX_NV] = v;
}

__global__ void processSuperNodes(
    unsigned int dyn_max_workload,
    unsigned int *glBuffers,
    unsigned int *d_global_queue,
    unsigned int *d_global_queue_tail,
    unsigned int *remain_workload,
    unsigned int local_WARPS_EACHBLK_dyn,
    unsigned int local_WARP_SIZE_dyn,
    G_pointers d_p,
    int level,
    int V)
{
    __shared__ unsigned int sh_dyn_max_workload;
    __shared__ unsigned int *glBuffer;
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int shBuffer_Tail;
    __shared__ unsigned int base_sh;
    __shared__ unsigned int shHead;
    __shared__ unsigned int share_v;

    unsigned int regTail;
    unsigned int i;

#ifndef inter_1_warp
    unsigned int WARPS_EACHBLK_dyn = 32;
    unsigned int WARP_SIZE_dyn = 32;
    unsigned int warp_id = THID >> 5;
    unsigned int lane_id = THID & 31;
#endif

#ifdef inter_1_warp
    unsigned int WARPS_EACHBLK_dyn = local_WARPS_EACHBLK_dyn;
    unsigned int WARP_SIZE_dyn = local_WARP_SIZE_dyn;
    unsigned int warp_id = THID / WARP_SIZE_dyn;
    unsigned int lane_id = THID & (WARP_SIZE_dyn - 1);
#endif

    if (THID == 0)
    {
        shHead = 0;
        shBuffer_Tail = 0;
        base_sh = 0;
        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE;
        sh_dyn_max_workload = dyn_max_workload;
    }
    __syncthreads();

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    while (true)
    {

        __syncthreads();

        if (shHead == d_global_queue_tail[0])
            break;

        if (THID == 0)
        {
            share_v = d_global_queue[shHead];
        }

        __syncthreads();

        for (unsigned int base = d_p.neighbors_offset[share_v + 1] - remain_workload[share_v]; base < d_p.neighbors_offset[share_v + 1]; base += N_THREADS)
        {
            unsigned int edge = base + global_threadIdx;

            if (edge >= d_p.neighbors_offset[share_v + 1])
                continue;

            unsigned int u = d_p.neighbors[edge];

            if (*(d_p.degrees + u) > level)
            {
                unsigned int a = atomicSub(d_p.degrees + u, 1);
                if (a == level + 1)
                {
                    unsigned int loc = atomicAdd(&shBuffer_Tail, 1);
                    writeTo_SM_Blk_Queue_scan(shBuffer, glBuffer, loc, u);
                }

                if (a == level)
                {
                    d_p.degrees[u] = level;
                }
            }
        }

        if (THID == 0)
            shHead += 1;

        while (true)
        {

            __syncthreads();

            if (base_sh == shBuffer_Tail)
                break;

#ifdef INTRA_1_CORE_warpdyn

            unsigned int remian_vertexs = shBuffer_Tail - base_sh;
            if (remian_vertexs <= 12)
            {

                if (remian_vertexs <= 6)
                {

                    if (remian_vertexs == 1)
                    {

                        WARP_SIZE_dyn = 1024;
                        WARPS_EACHBLK_dyn = 1;

                        warp_id = 0;
                        lane_id = THID;

                        // WARP_SIZE_dyn = 512;
                        // WARPS_EACHBLK_dyn = 2;

                        // warp_id = THID >> 9;
                        // lane_id = THID & 511;
                    }
                    else if (remian_vertexs == 2)
                    {
                        //    2>= x >1
                        WARP_SIZE_dyn = 512;
                        WARPS_EACHBLK_dyn = 2;

                        warp_id = THID >> 9;
                        lane_id = THID & 511;
                    }
                    else
                    { //    4>= x >3

                        //    6>= x >3
                        WARP_SIZE_dyn = 256;
                        WARPS_EACHBLK_dyn = 4;
                        warp_id = THID >> 8;
                        lane_id = THID & 255;
                    }
                }
                else
                { // 16 >= x >6

                    if (remian_vertexs <= 12)
                    { //    8>= x > 4
                      //    12 >= x > 6
                        WARP_SIZE_dyn = 128;
                        WARPS_EACHBLK_dyn = 8;

                        warp_id = THID >> 7;
                        lane_id = THID & 127;
                    }
                    else
                    { //    16>= x > 8
                      //   16>= x > 12
                        WARP_SIZE_dyn = 64;
                        WARPS_EACHBLK_dyn = 16;
                        warp_id = THID >> 6;
                        lane_id = THID & 63;
                    }
                }
            }

            else if (WARP_SIZE_dyn != local_WARP_SIZE_dyn)
            {
                WARP_SIZE_dyn = local_WARP_SIZE_dyn;
                WARPS_EACHBLK_dyn = local_WARPS_EACHBLK_dyn;
                warp_id = THID / WARP_SIZE_dyn;
                lane_id = THID & (WARP_SIZE_dyn - 1);
            }

#endif
            i = base_sh + warp_id;
            regTail = shBuffer_Tail;
            __syncthreads();

#ifndef SPlit_1_Vertex
            if (i >= regTail)
                continue;
#endif

            if (THID == 0)
            {
                base_sh += WARPS_EACHBLK_dyn;

                if (regTail < base_sh)
                    base_sh = regTail;
            }

            unsigned int start = 0;
            unsigned int end = 0;
            unsigned int v = 0;

            if (i < regTail)
            {
                v = readFrom_SM_Blk_Queue(shBuffer, glBuffer, i);

                start = d_p.neighbors_offset[v];
                end = d_p.neighbors_offset[v + 1];
            }
#ifdef SPlit_1_Vertex
            unsigned int workload_ = end - start;

            if (workload_ > sh_dyn_max_workload)
            {

                unsigned int new_start = end - remain_workload[v];
                if (remain_workload[v] > 0)
                {
                    start = new_start;

                    if (remain_workload[v] >= sh_dyn_max_workload)
                    {
                        if (remian_vertexs != 1 || remain_workload[v] >= MAXNUM_THREADS)
                        {
                            end = new_start + sh_dyn_max_workload;
                        }
                    }
                }
            }

            __syncthreads();
#endif

            while (true)
            {

                if (start >= end)
                    break;
                // 0 - WARP_SIZE
                unsigned int j = start + lane_id;
                // 下次处理 相隔 WARP_SIZE的节点
                start += WARP_SIZE_dyn;
                // 出界 不处理
                if (j >= end)
                    continue;

                unsigned int u = d_p.neighbors[j];
                if (*(d_p.degrees + u) > level)
                {
                    // 邻接点度数 大于 level 减一

                    unsigned int a = atomicSub(d_p.degrees + u, 1);
                    // 得到新的K-shell 节点 加入到 这个 block的 缓冲区中
                    if (a == level + 1)
                    {

                        unsigned int loc = atomicAdd(&shBuffer_Tail, 1);

                        writeTo_SM_Blk_Queue_scan(shBuffer, glBuffer, loc, u);

                        // writeToShBuffer(shBuffer, loc, u);

                        // if (*(d_p.degrees + u) >= MAXNUM_THREADS)
                        // {

                        //     printf("NEW SUPER FRONTIERS \n ");
                        // }
                    }

                    // 恢复到 level 水平
                    if (a == level)
                    {
                        // node degree became less than the level after decrementing...
                        // atomicAdd(d_p.degrees+u, 1);
                        d_p.degrees[u] = level;
                    }
                }
            }

// 大负载节点 少了  MAX_2_WORKLOAD条边
#ifdef SPlit_1_Vertex
            __syncthreads();

            // 判定是大负载节点
            // 判断为有效节点线程

            if (workload_ > sh_dyn_max_workload)
            {

                if (lane_id == 0)
                {
                    // 目前的end 是 小于 原来end ：  有剩余 边未处理
                    // 处理的是最后一拨边
                    if (end == d_p.neighbors_offset[v + 1])
                    {
                        remain_workload[v] = 0;
                    }
                    else
                    { // 只处理了一部分

                        //                 #ifdef GLOBAL_2_QUEUE
                        //                         if (remain_workload[v] >= 10240)
                        //                         {
                        //                             unsigned int loc = atomicAdd(d_global_queue_tail, 1);
                        //                             writeToBuffer_loop(d_global_queue, loc, v);
                        //                             //atomicAdd(&split_times, 1);
                        //                             remain_workload[v] -= sh_dyn_max_workload;

                        //                             // printf("level: %d, super big vertex: %u degree: %u tail: %d\n", level, v, d_p.neighbors_offset[v + 1] - d_p.neighbors_offset[v], d_global_queue_tail[0]);
                        //                         }
                        //                         else
                        // #endif

                        {
                            unsigned int loc = atomicAdd(&shBuffer_Tail, 1);
                            writeTo_SM_Blk_Queue_scan(shBuffer, glBuffer, loc, v);
                            // writeToShBuffer(shBuffer, loc, v);

                            remain_workload[v] -= sh_dyn_max_workload;
                        }
                    }
                }
            }

#endif
        }
    }
}


__global__ void Search_FirstFrontier_small_graph(
    unsigned int *smalltail,
    unsigned int *smallgraph,
    unsigned int *degrees,
    unsigned int level,
    unsigned int V)
{

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int base = 0; base < V; base += N_THREADS)
    {
        unsigned int v = base + global_threadIdx;

        if (v < V && degrees[v] >= level)
        {
            unsigned int t = atomicAdd(smalltail, 1);
            smallgraph[t] = v;
        }
    }
}

__global__ void Search_FirstFrontier(

    unsigned int *g_min_level,
    unsigned int dyn_max_workload,

#ifdef GLOBAL_2_QUEUE
    unsigned int *d_global_queue,
    unsigned int *d_global_queue_tail,
#endif

    unsigned int *remain_workload,
    unsigned int local_WARPS_EACHBLK_dyn, unsigned int local_WARP_SIZE_dyn,
    unsigned int *glBuffers, unsigned int *flag, G_pointers d_p, unsigned int *degrees,
    unsigned int level, unsigned int V,
    unsigned int *global_count)
{
    __shared__ unsigned int share_min;
    __shared__ unsigned int sh_dyn_max_workload;
    __shared__ unsigned int *glBuffer;
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
#ifdef SPlit_1_Vertex
    __shared__ unsigned int split_times;
#endif

#ifndef inter_1_warp
    unsigned int WARPS_EACHBLK_dyn = 32;
    unsigned int WARP_SIZE_dyn = 32;
    unsigned int warp_id = THID >> 5;
    unsigned int lane_id = THID & 31;
#endif

#ifdef inter_1_warp
    unsigned int WARPS_EACHBLK_dyn = local_WARPS_EACHBLK_dyn;
    unsigned int WARP_SIZE_dyn = local_WARP_SIZE_dyn;
    unsigned int warp_id = THID / WARP_SIZE_dyn;
    unsigned int lane_id = THID & (WARP_SIZE_dyn - 1);
#endif

    unsigned int regTail;
    unsigned int i;

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (THID == 0)
    {
        share_min = 0xFFFFFFFF;
        base = 0;
        bufTail = 0;
#ifdef SPlit_1_Vertex
        split_times = 0;
#endif
        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE;
        sh_dyn_max_workload = dyn_max_workload;
    }
    __syncthreads();

    for (unsigned int basev = 0; basev < V; basev += N_THREADS)
    {

        unsigned int v = basev + global_threadIdx;

        if (v >= V)
            continue;
        // TODO 感觉可以减少原子操作

        unsigned int degree_max = degrees[v];
        if (degree_max == level)
        {
            if (atomicAdd(flag + v, 1) == 0)
            {
                unsigned int loc = atomicAdd(&bufTail, 1);

                // writeToSMQueue_scan1(shBuffer, loc, v);

                writeTo_SM_Blk_Queue_scan(shBuffer, glBuffer, loc, v);
            }

            if (share_min != level)
                atomicMin(&share_min, degree_max);
        }

        if (degree_max > level)
        {
            atomicMin(&share_min, degree_max);
        }
    }

    while (true)
    {
        __syncthreads();
        // 情况1：所有warp都没有获得frontiers
        if (base == bufTail)
            break; // all the threads will evaluate to true at same iteration
                   // 情况2：至少有一个warp获得节点
                   // // syncthreads must be executed by all the threads

#ifdef INTRA_1_CORE_warpdyn

        unsigned int remian_vertexs = bufTail - base;
        if (remian_vertexs <= 16)
        {

            if (remian_vertexs <= 6)
            {

                if (remian_vertexs == 1)
                {

                    WARP_SIZE_dyn = 1024;
                    WARPS_EACHBLK_dyn = 1;

                    warp_id = 0;
                    lane_id = THID;
                }
                else if (remian_vertexs == 2)
                {
                    //    2>= x >1
                    WARP_SIZE_dyn = 512;
                    WARPS_EACHBLK_dyn = 2;

                    warp_id = THID >> 9;
                    lane_id = THID & 511;
                }
                else
                { //    4>= x >3

                    //    6>= x >3
                    WARP_SIZE_dyn = 256;
                    WARPS_EACHBLK_dyn = 4;

                    warp_id = THID >> 8;
                    lane_id = THID & 255;
                }
            }
            else
            { // 16 >= x >6

                if (remian_vertexs <= 12)
                { //    8>= x > 4
                  //    12 >= x > 6
                    WARP_SIZE_dyn = 128;
                    WARPS_EACHBLK_dyn = 8;

                    warp_id = THID >> 7;
                    lane_id = THID & 127;
                }
                else
                { //    16>= x > 8
                  //   16>= x > 12
                    WARP_SIZE_dyn = 64;
                    WARPS_EACHBLK_dyn = 16;
                    warp_id = THID >> 6;
                    lane_id = THID & 63;
                }
            }
        }

        else if (WARP_SIZE_dyn != local_WARP_SIZE_dyn)
        {
            WARP_SIZE_dyn = local_WARP_SIZE_dyn;
            WARPS_EACHBLK_dyn = local_WARPS_EACHBLK_dyn;
            warp_id = THID / WARP_SIZE_dyn;
            lane_id = THID & (WARP_SIZE_dyn - 1);
        }

#endif

        i = base + warp_id;
        // 情况2：更新目前队列的真实队尾
        regTail = bufTail;

        __syncthreads();
        // 情况2：更新目前队列的真实队尾后，超过队尾的warp等待

#ifndef SPlit_1_Vertex
        if (i >= regTail)
            continue;
#endif

        // 情况2：试探 队头 + WARPS_EACHBLK_dyn, 如果队头大于 真实队尾 就保持不变

        if (THID == 0)
        {
            base += WARPS_EACHBLK_dyn;

            // intra_iter %= 512;
            // intra_iter+=1;
            if (regTail < base)
                base = regTail;
        }
        // bufTail is incremented in the code below:
        //  一个warp处理 一个节点的所有邻接点
        unsigned int start = 0;
        unsigned int end = 0;
        unsigned int v = 0;

        if (i < regTail)
        {

            v = readFrom_SM_Blk_Queue(shBuffer, glBuffer, i);
            start = d_p.neighbors_offset[v];
            end = d_p.neighbors_offset[v + 1];
        }
#ifdef SPlit_1_Vertex
        unsigned int workload_ = end - start;

        if (workload_ > sh_dyn_max_workload)
        {
            // 更新大负载节点的 边终点位置  只计算  dyn_max_workload个 边
            // TODO:  多个warp在本轮都访问该值 会出现错误

            // 新的开始： end- remain_workload;
            // remain_workload : 0 ~ degree;

            unsigned int new_start = end - remain_workload[v];

            //  新的结束： warp需要进行试探  dyn_max_workload个边去处理 ：
            // unsigned int new_end = new_start + sh_dyn_max_workload;

            // TODO: 应该能保证 剩余workload 大于0
            if (remain_workload[v] > 0)
            {
                //  printf("v: %u workload: %u \n", v,workload_);
                start = new_start;
                // 1： 剩余大于  sh_dyn_max_workload  等于原end
                if (remain_workload[v] >= sh_dyn_max_workload)
                {
                    if (remian_vertexs != 1 || remain_workload[v] >= MAXNUM_THREADS)
                    {
                        end = new_start + sh_dyn_max_workload;
                    }
                }

                // 2： 剩余很少小于  sh_dyn_max_workload  等于原end
                // 省略 end = end;
            }

            //     printf("new %d\n ", new_start);

            //    printf("end %d\n ", new_start);
        }

        __syncthreads();
#endif
        // WARP_SIZE 个线程 处理一个节点的邻接点
        // 如 线程0 会处理 第0 WARP_SIZE  dyn_max_workload的邻接点

        while (true)
        {
            // __syncwarp(0);
            // warp 工作列表退出
            if (start >= end)
                break;
            // 0 - WARPSIZE_dyn
            unsigned int j = start + lane_id;
            // 下次处理 相隔 WARPSIZE_dyn的节点
            start += WARP_SIZE_dyn;
            // 出界 不处理
            if (j >= end)
                continue;

            unsigned int u = d_p.neighbors[j];
            if (*(d_p.degrees + u) > level)
            {
                // 邻接点度数 大于 level 减一

                unsigned int a = atomicSub(d_p.degrees + u, 1);
                // 得到新的K-shell 节点 加入到 这个 block的 缓冲区中
                if (a == level + 1)
                {
                    if (atomicAdd(flag + u, 1) == 0)
                    {
                        // printf(" add1 \n");
                        unsigned int loc = atomicAdd(&bufTail, 1);
                        writeTo_SM_Blk_Queue_loop(shBuffer, glBuffer, loc, u);
                        // writeToSMQueue_loop1(shBuffer, loc, u);
                    }
                }

                // 恢复到 level 水平
                if (a == level)
                {
                    d_p.degrees[u] = level;
                }
            }
        }

#ifdef SPlit_1_Vertex
        __syncthreads();

        // 判定是大负载节点
        // 判断为有效节点线程

        if (workload_ > sh_dyn_max_workload)
        {

            if (lane_id == 0)
            {
                // 目前的end 是 小于 原来end ：  有剩余 边未处理
                if (end <= d_p.neighbors_offset[v + 1])
                {

                    //  printf("splite node:  %u,   remian %u , dregree %u \n", v, remain_workload[v], d_p.neighbors_offset[v + 1] - d_p.neighbors_offset[v] );
                    //  printf("end  %d\n ", d_p.neighbors_offset[v + 1] - end);
                    // 处理的是最后一拨边
                    if (end == d_p.neighbors_offset[v + 1])
                    {
                        remain_workload[v] = 0;
                    }
                    else
                    { // 只处理了一部分
#ifdef GLOBAL_2_QUEUE
                        if (remain_workload[v] >= MAXNUM_THREADS)
                        {
                            unsigned int loc = atomicAdd(d_global_queue_tail, 1);
                            writeToBuffer_loop(d_global_queue, loc, v);
                            atomicAdd(&split_times, 1);
                            remain_workload[v] -= sh_dyn_max_workload;

                            // printf("level: %d, super big vertex: %u degree: %u tail: %d\n", level, v, d_p.neighbors_offset[v + 1] - d_p.neighbors_offset[v], d_global_queue_tail[0]);
                        }
                        else
#endif
                        {
                            unsigned int loc = atomicAdd(&bufTail, 1);
                            writeTo_SM_Blk_Queue_loop(shBuffer, glBuffer, loc, v);

                            atomicAdd(&split_times, 1);

                            remain_workload[v] -= sh_dyn_max_workload;
                        }
                    }
                }
            }
        }

#endif
    }

    if (THID == 0 && share_min != 0xFFFFFFFF)
    {
        // printf(" ############### share_min  %u   \n", share_min);
        atomicMin(g_min_level, share_min);
    }

    if (THID == 0 && bufTail > 0)
    {

#ifdef SPlit_1_Vertex
        bufTail -= split_times;
#endif
        atomicAdd(global_count, bufTail);
    }
}

__global__ void Search_FirstFrontier_from_smallgraph(

    unsigned int *g_min_level,

    unsigned int dyn_max_workload,
#ifdef GLOBAL_2_QUEUE
    unsigned int *d_global_queue,
    unsigned int *d_global_queue_tail,
#endif

    unsigned int *remain_workload,
    unsigned int local_WARPS_EACHBLK_dyn, unsigned int local_WARP_SIZE_dyn,

    unsigned int *smalltail,
    unsigned int *smallgraph, unsigned int *glBuffers,
    unsigned int *flag, G_pointers d_p,
    unsigned int *degrees, unsigned int level, unsigned int V)
{

    __shared__ unsigned int share_min;
    __shared__ unsigned int sh_dyn_max_workload;
    __shared__ unsigned int *glBuffer;
    __shared__ unsigned int shBuffer[MAX_NV];
    __shared__ unsigned int bufTail;

    __shared__ unsigned int base;

#ifndef inter_1_warp
    unsigned int WARPS_EACHBLK_dyn = 32;
    unsigned int WARP_SIZE_dyn = 32;
    unsigned int warp_id = THID >> 5;
    unsigned int lane_id = THID & 31;
#endif

#ifdef inter_1_warp
    unsigned int WARPS_EACHBLK_dyn = local_WARPS_EACHBLK_dyn;
    unsigned int WARP_SIZE_dyn = local_WARP_SIZE_dyn;
    unsigned int warp_id = THID / WARP_SIZE_dyn;
    unsigned int lane_id = THID & (WARP_SIZE_dyn - 1);
#endif

    unsigned int regTail;
    unsigned int i;

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (THID == 0)
    {
        share_min = 0xFFFFFFFF;
        base = 0;
        bufTail = 0;

        glBuffer = glBuffers + blockIdx.x * GLBUFFER_SIZE;

        sh_dyn_max_workload = dyn_max_workload;
    }

    for (unsigned int base = 0; base < smalltail[0]; base += N_THREADS)
    {
        unsigned int index = base + global_threadIdx;
        if (index >= smalltail[0])
            continue;

        unsigned int v = smallgraph[index];
        unsigned int degree_max = degrees[v];

        if (degree_max == level)
        {

            if (atomicAdd(flag + v, 1) == 0)
            {
                unsigned int loc = atomicAdd(&bufTail, 1);
                writeTo_SM_Blk_Queue_scan(shBuffer, glBuffer, loc, v);
            }

            if (share_min != level)
                atomicMin(&share_min, degree_max);
        }

        if (share_min != level && degree_max > level)
        {
            atomicMin(&share_min, degree_max);
        }
    }

    __syncthreads();

    if (THID == 0 && share_min != 0xFFFFFFFF)
    {
        // printf(" ############### share_min  %u   \n", share_min);
        atomicMin(g_min_level, share_min);
    }

    while (true)
    {
        __syncthreads();
        // 情况1：所有warp都没有获得frontiers
        if (base == bufTail)
            break;

#ifdef INTRA_1_CORE_warpdyn

        unsigned int remian_vertexs = bufTail - base;
        if (remian_vertexs <= 16)
        {

            if (remian_vertexs <= 6)
            {

                if (remian_vertexs == 1)
                {

                    WARP_SIZE_dyn = 1024;
                    WARPS_EACHBLK_dyn = 1;

                    warp_id = 0;
                    lane_id = THID;
                }
                else if (remian_vertexs == 2)
                {
                    //    2>= x >1
                    WARP_SIZE_dyn = 512;
                    WARPS_EACHBLK_dyn = 2;

                    warp_id = THID >> 9;
                    lane_id = THID & 511;
                }
                else
                { //    4>= x >3

                    //    6>= x >3
                    WARP_SIZE_dyn = 256;
                    WARPS_EACHBLK_dyn = 4;

                    warp_id = THID >> 8;
                    lane_id = THID & 255;
                }
            }
            else
            { // 16 >= x >6

                if (remian_vertexs <= 12)
                { //    8>= x > 4
                  //    12 >= x > 6
                    WARP_SIZE_dyn = 128;
                    WARPS_EACHBLK_dyn = 8;

                    warp_id = THID >> 7;
                    lane_id = THID & 127;
                }
                else
                { //    16>= x > 8
                  //   16>= x > 12
                    WARP_SIZE_dyn = 64;
                    WARPS_EACHBLK_dyn = 16;
                    warp_id = THID >> 6;
                    lane_id = THID & 63;
                }
            }
        }
        else if (WARP_SIZE_dyn != local_WARP_SIZE_dyn)
        {
            WARP_SIZE_dyn = local_WARP_SIZE_dyn;
            WARPS_EACHBLK_dyn = local_WARPS_EACHBLK_dyn;
            warp_id = THID / WARP_SIZE_dyn;
            lane_id = THID & (WARP_SIZE_dyn - 1);
        }

#ifdef PrintF2
        if (bufTail - base <= 2048 && THID == 0 && blockIdx.x <= 0)
        {
            printf(" \n blockID %u, bufftail - base %u  $$ WARP_SIZE_dyn %d  level %d $$ \n \n", blockIdx.x, bufTail - base, WARP_SIZE_dyn, level);
        }
#endif

#endif

        i = base + warp_id;
        // 情况2：更新目前队列的真实队尾
        regTail = bufTail;
        __syncthreads();
        // 情况2：更新目前队列的真实队尾后，超过队尾的warp等待

#ifndef SPlit_1_Vertex
        if (i >= regTail)
            continue;
#endif

        // 情况2：试探 队头 + WARPS_EACHBLK_dyn, 如果队头大于 真实队尾 就保持不变

        if (THID == 0)
        {
            base += WARPS_EACHBLK_dyn;
            // intra_iter %= 512;
            // intra_iter+=1;
            if (regTail < base)
                base = regTail;
        }
        // bufTail is incremented in the code below:
        //  一个warp处理 一个节点的所有邻接点
        unsigned int start = 0;
        unsigned int end = 0;
        unsigned int v = 0;

        // if (i >= regTail)
        // {

        //     v = 0;
        // }
        // else
        if (i < regTail)
        {

            v = readFrom_SM_Blk_Queue(shBuffer, glBuffer, i);
            start = d_p.neighbors_offset[v];
            end = d_p.neighbors_offset[v + 1];
        }
#ifdef SPlit_1_Vertex
        unsigned int workload_ = end - start;
        // if(workload_ ==  25781){
        //     printf("vertex %d\n ", v);
        // }
        // unsigned int  _todo_workload_start;

        // if(THID==0 && v==0)  printf("v: %u workload: %u \n", v,workload_);
        //  没有分配节点的warp workload为0 自然小于  sh_dyn_max_workload+WARP_SIZE_dyn
        // 该节点workload_ 大等于 sh_dyn_max_workload 就准备切分

        if (workload_ > sh_dyn_max_workload)
        {
            // 更新大负载节点的 边终点位置  只计算  dyn_max_workload个 边
            // TODO:  多个warp在本轮都访问该值 会出现错误

            // 新的开始： end- remain_workload;
            // remain_workload : 0 ~ degree;

            unsigned int new_start = end - remain_workload[v];

            //  新的结束： warp需要进行试探  dyn_max_workload个边去处理 ：
            // unsigned int new_end = new_start + sh_dyn_max_workload;

            // TODO: 应该能保证 剩余workload 大于0
            if (remain_workload[v] > 0)
            {
                //  printf("v: %u workload: %u \n", v,workload_);
                start = new_start;
                // 1： 剩余大于  sh_dyn_max_workload  等于原end
                if (remain_workload[v] >= sh_dyn_max_workload)
                {
                    if (remian_vertexs != 1 || remain_workload[v] >= MAXNUM_THREADS)
                    {
                        end = new_start + sh_dyn_max_workload;
                    }
                }

                // 2： 剩余很少小于  sh_dyn_max_workload  等于原end
                // 省略 end = end;
            }

            //     printf("new %d\n ", new_start);

            //    printf("end %d\n ", new_start);
        }

        __syncthreads();
#endif
        // WARP_SIZE 个线程 处理一个节点的邻接点
        // 如 线程0 会处理 第0 WARP_SIZE  dyn_max_workload的邻接点

        while (true)
        {
            // __syncwarp(0);
            // warp 工作列表退出
            if (start >= end)
                break;
            // 0 - WARPSIZE_dyn
            unsigned int j = start + lane_id;
            // 下次处理 相隔 WARPSIZE_dyn的节点
            start += WARP_SIZE_dyn;
            // 出界 不处理
            if (j >= end)
                continue;

            unsigned int u = d_p.neighbors[j];
            if (*(d_p.degrees + u) > level)
            {
                // 邻接点度数 大于 level 减一

                unsigned int a = atomicSub(d_p.degrees + u, 1);
                // 得到新的K-shell 节点 加入到 这个 block的 缓冲区中
                if (a == level + 1)
                {
                    if (atomicAdd(flag + u, 1) == 0)
                    {
                        // printf(" add1 \n");
                        unsigned int loc = atomicAdd(&bufTail, 1);
                        writeTo_SM_Blk_Queue_loop(shBuffer, glBuffer, loc, u);
                        // writeToSMQueue_loop1(shBuffer, loc, u);
                    }
                }

                // 恢复到 level 水平
                if (a == level)
                {
                    d_p.degrees[u] = level;
                }
            }
        }

#ifdef SPlit_1_Vertex
        __syncthreads();

        // 判定是大负载节点
        // 判断为有效节点线程

        if (workload_ > sh_dyn_max_workload)
        {

            if (lane_id == 0)
            {
                // 目前的end 是 小于 原来end ：  有剩余 边未处理
                if (end <= d_p.neighbors_offset[v + 1])
                {

                    //  printf("splite node:  %u,   remian %u , dregree %u \n", v, remain_workload[v], d_p.neighbors_offset[v + 1] - d_p.neighbors_offset[v] );
                    //  printf("end  %d\n ", d_p.neighbors_offset[v + 1] - end);
                    // 处理的是最后一拨边
                    if (end == d_p.neighbors_offset[v + 1])
                    {
                        remain_workload[v] = 0;
                    }
                    else
                    {

#ifdef GLOBAL_2_QUEUE
                        if (remain_workload[v] >= MAXNUM_THREADS)
                        {
                            unsigned int loc = atomicAdd(d_global_queue_tail, 1);
                            writeToBuffer_loop(d_global_queue, loc, v);
                            remain_workload[v] -= sh_dyn_max_workload;

                            // printf("level: %d, super big vertex: %u degree: %u tail: %d\n", level, v, d_p.neighbors_offset[v + 1] - d_p.neighbors_offset[v], d_global_queue_tail[0]);
                        }
                        else
#endif

                        {
                            unsigned int loc = atomicAdd(&bufTail, 1);
                            writeTo_SM_Blk_Queue_loop(shBuffer, glBuffer, loc, v);

                            remain_workload[v] -= sh_dyn_max_workload;
                        }
                    }
                }
            }
        }

#endif
    }
}

int AdaptiveCore(Graph &data_graph)
{

    G_pointers data_pointers;

    malloc_graph_gpu_memory(data_graph, data_pointers);

    unsigned int h_dyn_max_workload = 1024;

    unsigned int *h_remian_workload = NULL;
    chkerr(cudaMalloc(&h_remian_workload, sizeof(unsigned int) * data_graph.V));
    cudaMemcpy(h_remian_workload, data_pointers.degrees, data_graph.V * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int *global_tail;
    chkerr(cudaMalloc(&global_tail, sizeof(unsigned int)));
    cudaMemset(global_tail, 0, sizeof(unsigned int));

    unsigned int d_global_tail = 0;

    unsigned int *global_queue = NULL;
    chkerr(cudaMalloc(&global_queue, sizeof(unsigned int) * data_graph.V));
#ifdef GLOBAL_2_QUEUE
#endif

#ifdef JUMPCORE
    unsigned int h_level = 0;
    unsigned int full_level = FULL;
    unsigned int *g_min_level = NULL;

    chkerr(cudaMalloc(&g_min_level, sizeof(unsigned int)));
    cudaMemcpy(g_min_level, &full_level, sizeof(unsigned int), cudaMemcpyHostToDevice);

#endif

    unsigned int level = 0;

    unsigned int count = 0;
    unsigned int *global_count = NULL;

    unsigned int *flag = NULL;

    unsigned int *smallgraph = NULL;

    unsigned int *smalltail;

    chkerr(cudaMallocManaged(&smalltail, sizeof(unsigned int)));

    smalltail[0] = 0;

    chkerr(cudaMalloc(&smallgraph, sizeof(unsigned int) * data_graph.V));
    cudaMemset(smallgraph, 0, sizeof(unsigned int) * data_graph.V);

    chkerr(cudaMalloc(&flag, sizeof(unsigned int) * data_graph.V));
    cudaMemset(flag, 0, sizeof(unsigned int) * data_graph.V);

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    unsigned int *glBuffers = NULL;
  
    chkerr(cudaMalloc(&glBuffers, sizeof(unsigned int) * BLK_NUMS * GLBUFFER_SIZE));

#ifndef inter_1_warp
    unsigned int WARPSIZE_dyn = 32;
#endif

#ifdef inter_1_warp
    unsigned int WARPSIZE_dyn = 1;
#endif

    unsigned int WARPSEACHBLK_dyn = BLK_DIM / WARPSIZE_dyn;

    auto start_ = chrono::steady_clock::now();

    while (true)
    {

#ifdef inter_1_warp
        // TODO: 因为k不是连续递增的
        if (level > WARPSIZE_dyn && level <= 128)
        {

            WARPSIZE_dyn *= 2;
            WARPSEACHBLK_dyn = BLK_DIM / WARPSIZE_dyn;
            h_dyn_max_workload = 2048;
        }
        else if (level > 128)
        {

            // WARPSIZE_dyn *= 2;
            WARPSIZE_dyn = 64;
            WARPSEACHBLK_dyn = BLK_DIM / WARPSIZE_dyn;
            h_dyn_max_workload = 2048;

            if (level >= 1024)
            {
                h_dyn_max_workload = 4096;
            }
        }

#endif

        if (count >= data_graph.V * SMALL_PERCENT)
        {
            break;
        }

        cudaMemcpy(g_min_level, &full_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
        Search_FirstFrontier<<<BLK_NUMS, BLK_DIM>>>(
            g_min_level,
            h_dyn_max_workload,
#ifdef GLOBAL_2_QUEUE
            global_queue,
            global_tail,

#endif

            h_remian_workload,
            WARPSEACHBLK_dyn, WARPSIZE_dyn, glBuffers, flag, data_pointers, data_pointers.degrees, level,
            data_graph.V, global_count);

        chkerr(cudaMemcpy(&h_level, g_min_level, sizeof(unsigned int), cudaMemcpyDeviceToHost));

#ifdef GLOBAL_2_QUEUE

        if (h_level == level)
        {
            chkerr(cudaMemcpy(&d_global_tail, global_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (d_global_tail != 0)
            {
                processSuperNodes<<<BLK_NUMS, BLK_DIM>>>(
                    h_dyn_max_workload,
                    glBuffers,
                    global_queue,
                    global_tail,
                    h_remian_workload, WARPSEACHBLK_dyn, WARPSIZE_dyn, data_pointers, level, data_graph.V);
#ifdef GLOBAL_2_QUEUE
                cudaMemset(global_tail, 0, sizeof(unsigned int));
#endif
            }
        }

#endif
        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        if (h_level == level)
        {
            level++;
        }
        else
        {
            level = h_level;
        }
    }

    Search_FirstFrontier_small_graph<<<BLK_NUMS, BLK_DIM>>>(smalltail, smallgraph, data_pointers.degrees, level, data_graph.V);

    // cout << endl
    //      << "*********Now Level:" << level << endl;
    while (true)
    {

        // cout << endl
        //      << "*********Now Level:" <<level<< endl;
//  break;
#ifdef inter_1_warp

        if (level > WARPSIZE_dyn && level <= 128)
        {

            WARPSIZE_dyn *= 2;
            WARPSEACHBLK_dyn = BLK_DIM / WARPSIZE_dyn;
            h_dyn_max_workload = 2048;
        }
        else if (level > 128)
        {

            // WARPSIZE_dyn *= 2;
            WARPSIZE_dyn = 64;
            WARPSEACHBLK_dyn = BLK_DIM / WARPSIZE_dyn;

            h_dyn_max_workload = 2048;

            if (level >= 1024)
            {
                h_dyn_max_workload = 4096;
            }
        }

#endif

        // while (true)
        // {

        cudaMemcpy(g_min_level, &full_level, sizeof(unsigned int), cudaMemcpyHostToDevice);

        Search_FirstFrontier_from_smallgraph<<<BLK_NUMS, BLK_DIM>>>(
            g_min_level,

            h_dyn_max_workload,
#ifdef GLOBAL_2_QUEUE
            global_queue,
            global_tail,

#endif

            h_remian_workload,
            WARPSEACHBLK_dyn, WARPSIZE_dyn, smalltail, smallgraph, glBuffers, flag, data_pointers, data_pointers.degrees, level,
            data_graph.V);

        chkerr(cudaMemcpy(&h_level, g_min_level, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //   cout << " h_level: " << h_level <<" level: " <<level<<endl;
        if (h_level == FULL)
        {
            break;
        }

#ifdef GLOBAL_2_QUEUE

        if (h_level == level)
        {
            chkerr(cudaMemcpy(&d_global_tail, global_tail, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (d_global_tail != 0)
            {
                processSuperNodes<<<BLK_NUMS, BLK_DIM>>>(
                    h_dyn_max_workload,
                    glBuffers,
                    global_queue,
                    global_tail,
                    h_remian_workload, WARPSEACHBLK_dyn, WARPSIZE_dyn, data_pointers, level, data_graph.V);

#ifdef GLOBAL_2_QUEUE
                cudaMemset(global_tail, 0, sizeof(unsigned int));
#endif
            }
        }

#endif

        if (h_level == level)
        {
            level++;
        }
        else
        {
            level = h_level;
        }
    }

    auto end = chrono::steady_clock::now();
    data_graph.kmax = level - 1;

    //  cout << "WARPSIZE_dyn: " << WARPSIZE_dyn << endl;

    cudaFree(global_queue);
#ifdef GLOBAL_2_QUEUE
#endif
    cudaFree(h_remian_workload);
    cudaFree(smallgraph);
    cudaFree(flag);
    cudaFree(glBuffers);
    //  cudaFree(bufTails);
    free_graph_gpu_memory(data_pointers);

    return chrono::duration_cast<chrono::milliseconds>(end - start_).count();
}
