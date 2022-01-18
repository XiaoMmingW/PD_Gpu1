//
// Created by wxm on 2021/12/2.
//

#ifndef PD_PARALLEL_SURF_CORR_CUH
#define PD_PARALLEL_SURF_CORR_CUH
#include "Base.cuh"
void surface_correct_gpu(Dselect &d, BAtom &ba, BBond &bb);
void surface_correct_cpu(Dselect &d, BAtom &ba, BBond &bb);
__global__ void kernel_vol_Corr
(int *NN, int *NL, real *x, real *idist, real *fac, real *y=NULL, real *z=NULL);
void vol_Corr(int *NN, int *NL, real *x, real *idist, real *fac, real *y=NULL, real *z=NULL);
#endif //PD_PARALLEL_SURF_CORR_CUH
