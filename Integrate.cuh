//
// Created by wxm on 2021/12/3.
//

#ifndef PD_PARALLEL_INTEGRATE_CUH
#define PD_PARALLEL_INTEGRATE_CUH
#include "Base.cuh"
#include "Reduce.cuh"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
//准静态积分
void static_integrate_gpu(Dselect &d, BAtom &ba, int ct=0);
__global__ void kernel_cnn(
        real *cn_xy, real *pforce_x, real *pforceold_x,   real *velhalfold_x,
        real *disp_x, real *disp_xy, real *pforce_y=NULL, real *pforceold_y=NULL,
        real *velhalfold_y=NULL, real *disp_y=NULL, real *pforce_z=NULL, real *pforceold_z=NULL,
        real *velhalfold_z=NULL, real *disp_z=NULL);
real cal_cn( BAtom &ba);
__global__ void kernel_integrate(
        int ct,real cn,  real *pforce_x, real *velhalf_x, real *bforce_x,
        real *vel_x,  real *velhalfold_x,  real *disp_x, real *pforceold_x, real *pforce_y=NULL,
        real *velhalf_y=NULL, real *bforce_y=NULL,  real *vel_y=NULL, real *velhalfold_y=NULL,
        real *disp_y=NULL,  real *pforceold_y=NULL, real *pforce_z=NULL, real *velhalf_z=NULL, real *bforce_z=NULL,
        real *vel_z=NULL, real *velhalfold_z=NULL, real *disp_z=NULL,   real *pforceold_z=NULL);

void static_integrate_cpu(
        int ct, real *pforce_x, real *velhalf_x, real *bforce_x,
        real *vel_x,  real *velhalfold_x,  real *disp_x, real *pforceold_x, real *pforce_y=NULL,
        real *velhalf_y=NULL, real *bforce_y=NULL,  real *vel_y=NULL, real *velhalfold_y=NULL,
        real *disp_y=NULL,  real *pforceold_y=NULL, real *pforce_z=NULL, real *velhalf_z=NULL, real *bforce_z=NULL,
        real *vel_z=NULL, real *velhalfold_z=NULL, real *disp_z=NULL,   real *pforceold_z=NULL);

//显示积分
__global__ void kernel_integrate_2D_1(real *vel_x, real *acc_x,  real *disp_x, real *vel_y, real *acc_y, real *disp_y);
__global__ void  kernel_integrate_3D_1(real *vel_x, real *acc_x,  real *disp_x, real *vel_y, real *acc_y, real *disp_y,
        real *disp_z, real *vel_z, real *acc_z);
__global__ void  kernel_integrate_2D_2(real *vel_x, real *acc_x, real *pforce_x, real *bforce_x, real *vel_y,
                                       real *acc_y,  real *pforce_y, real *bforce_y);
__global__ void  kernel_integrate_3D_2(
        real *vel_x, real *acc_x, real *pforce_x, real *bforce_x,
        real *vel_y, real *acc_y,  real *pforce_y, real *bforce_y,
        real *vel_z, real *acc_z, real *pforce_z, real *bforce_z);
void integrate_1(real *vel_x, real *acc_x,  real *disp_x, real *vel_y, real *acc_y, real *disp_y,
                        real *vel_z=NULL, real *acc_z=NULL, real *disp_z=NULL);
void integrate_2(real *vel_x, real *acc_x, real *pforce_x, real *bforce_x,
                        real *vel_y, real *acc_y,  real *pforce_y, real *bforce_y,
                        real *vel_z=NULL, real *acc_z=NULL, real *pforce_z=NULL, real *bforce_z=NULL);
real  cal_cn_cpu(real *pforce_x,   real *velhalfold_x,  real *disp_x,
                 real *pforceold_x, real *pforce_y, real *velhalfold_y,
                 real *disp_y,  real *pforceold_y, real *pforce_z=NULL,
                 real *velhalfold_z=NULL, real *disp_z=NULL,   real *pforceold_z=NULL);
real  cal_cn2_cpu(real *cn_xy, real *disp_xy);
void cal_cn3_cpu(real *cn_xy,real *disp_xy, real *pforce_x,   real *velhalfold_x,  real *disp_x,
                 real *pforceold_x, real *pforce_y, real *velhalfold_y,
                 real *disp_y,  real *pforceold_y,real *pforce_z=NULL,
                 real *velhalfold_z=NULL, real *disp_z=NULL,   real *pforceold_z=NULL);
#endif //PD_PARALLEL_INTEGRATE_CUH
