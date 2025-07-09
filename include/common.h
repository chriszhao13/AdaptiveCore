
#ifndef CUTS_COMMON_H
#define CUTS_COMMON_H
#define MAXNUM_THREADS 62976
#define BLK_NUMS 246
#define MAX_NV 12200
#define MAX_Histo 11840
#define GLBUFFER_SIZE 1000000
#define All_GLBUFFER_SIZE GLBUFFER_SIZE *BLK_NUMS
#define SMALL_PERCENT 0.99
#define WARP_SIZE 32
#define WARP_SIZE_minus_1 WARP_SIZE - 1
#define WARP_index 5

// #################################//
#define WARPS_EACH_BLK (BLK_DIM / WARP_SIZE)
#define WORK_UNITS (BLK_NUMS * WARPS_EACH_BLK)
#define BLK_DIM 1024
#define BUFF_SIZE 100000
#define N_THREADS (BLK_DIM * BLK_NUMS)

#define THID threadIdx.x

#define UINT unsigned int
#define DS_LOC string("")
#define OUTPUT_LOC string("./output/")
#define REP 10
#define LANEID (THID & WARP_SIZE_minus_1)
#define WARPID (THID >> WARP_index)
#define FULL 0xFFFFFFFF
#define MAX_PREF 180
#define BLKID blockIdx.x
#define N_WARPS (BLK_NUMS * WARPS_EACH_BLK)
#define GLWARPID (BLKID * WARPS_EACH_BLK + WARPID)
#define GTHID (BLKID * N_THREADS + THID)

#define RESET "\033[0m"
#define BLACK "\033[30m"   /* Black */
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[38m"   /* White */

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <map>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include <deque>
#include <random>
#include <bits/stdc++.h>
#include "omp.h"

using namespace std;

typedef struct G_pointers
{

    unsigned int *neighbors;

    unsigned int *neighbors_oom;
    unsigned int *neighbors_offset;
    unsigned int *degrees;
    unsigned int V;
} G_pointers; // graph related


typedef struct Node
{
    unsigned int data[BUFF_SIZE];
    unsigned int limit;
    Node *next;
    Node *prev;
} Node;

#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m"   /* Red */
#define GREEN "\033[32m" /* Green */

#endif // CUTS_COMMON_H
