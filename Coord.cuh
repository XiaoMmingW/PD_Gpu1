//
// Created by wxm on 2021/12/2.
//

#ifndef PD_PARALLEL_COORD_CUH
#define PD_PARALLEL_COORD_CUH
#include "Base.cuh"

void Gen_Coord_plate_gpu(BAtom &ba);
__global__ void kernel_Initial_coord_plate(real *x, real *y);
__global__ void kernel_coord_plate_crack(real *x, real *y);
void Initial_coord_plate_cpu(real *x, real *y);
void coord_plate_crack_cpu(real *x, real *y);
int coord_KW(real *x, real *y, real *z);
#endif //PD_PARALLEL_COORD_CUH
