//
// Created by wxm on 2021/12/4.
//

#ifndef PD_PARALLEL_FORCE_STATE_CUH
#define PD_PARALLEL_FORCE_STATE_CUH
#include "Base.cuh"
__global__ void kernel_initial_weight_2D(
        real *m, int *NN, real *w, real *idist, real *fac);
__global__ void kernel_initial_weight_3D(
        real *m, int *NN, real *w, real *idist, real *fac);
__global__ void kernel_cal_theta_2D(
        int *NN, real *m, real *theta, int *NL, real *w, int *fail, real *e, real *idist, real *fac,
        real *x, real *disp_x, real *y, real *disp_y);
__global__  void kernel_state_force_2D(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist,
        real *fac, real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y);
__global__ void kernel_cal_theta_3D(
        int *NN, real *m, real *theta, int *NL, real *w, int *fail, real *e, real *idist, real *fac,
        real *x, real *disp_x, real *y, real *disp_y, real *z, real *disp_z);
__global__  void kernel_state_force_3D(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist, real *fac,
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, real *z, real *disp_z, real *pforce_z);
void initial_weight_cpu(real *m, int *NN,  real *w, real *idist, real *fac);
void state_force_2D_cpu(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist,
        real *fac, real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y
        );
__global__ void kernel_cal_theta_2D_particle(
        int *NN, real *m, real *theta, int *NL, real *w, int *fail, real *e, real *idist, real *fac,
        real *x, real *disp_x, real *y, real *disp_y);
__global__  void kernel_state_force_2D_particle(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist,
        real *fac, real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y
        );
#endif //PD_PARALLEL_FORCE_STATE_CUH
