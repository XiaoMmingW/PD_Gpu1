//
// Created by wxm on 2021/12/4.
//

#ifndef PD_PARALLEL_FORCE_BOND_CUH
#define PD_PARALLEL_FORCE_BOND_CUH
#include "Base.cuh"

void bond_force_2D_gpu(BAtom &ba, BBond &bb);
void bond_force_3D_gpu(BAtom &ba, BBond &bb);
__global__  void kernel_bond_force_2D(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac);
void bond_force_2D_gpu(BAtom &ba, BBond &bb);
__global__  void kernel_bond_force_2D_2(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac);
__global__  void kernel_bond_force_3D(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, real *z, real *disp_z,
        real *pforce_z, int *NN, int *NL, real *idist, real *scr, int *fail, real *fac);
void bond_force_cpu(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac, real *z=NULL, real *disp_z=NULL, real *pfprce_z=NULL);

__global__  void kernel_particle_force_2D(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac);
#endif //PD_PARALLEL_FORCE_BOND_CUH
