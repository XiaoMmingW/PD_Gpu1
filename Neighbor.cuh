//
// Created by wxm on 2021/12/2.
//

#ifndef PD_PARALLEL_NEIGHBOR_CUH
#define PD_PARALLEL_NEIGHBOR_CUH
#include "Base.cuh"
void find_neighbor_gpu(Dselect &d, BAtom &ba, BBond &bb);
void find_neighbor_2D(const real *x, const real *y, int *NN, int *NL);
void find_neighbor_3D(const real *x, const real *y, const real *z, int *NN, int *NL);
__global__ void  kernel_find_neighbor_2D(const real *x, const real *y, int *NN, int *NL);
__global__ void  kernel_find_neighbor_3D(const real *x, const real *y, const real *z,int *NN, int *NL);
#endif //PD_PARALLEL_NEIGHBOR_CUH
