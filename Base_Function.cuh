//
// Created by wxm on 2021/12/2.
//

#ifndef PD_PARALLEL_BASE_FUNCTION_CUH
#define PD_PARALLEL_BASE_FUNCTION_CUH
#include "Base.cuh"
long double cpuSecond();
__global__ void kernel_set_crack_2D(
        real clength, real loc_x, real loc_y,real theta, int *NN, int *fail, int *NL, real *x, real *y);
void set_crack_2D_cpu(real clength, real loc_x, real loc_y,real theta, int *NN, int *fail, int *NL, real *x, real *y);
__global__ void kernel_cal_dmg_2D(int *NN,real *dmg, int *fail, real *fac);
__global__ void kernel_cal_dmg_3D(int *NN,real *dmg, int *fail, real *fac);
void cal_dmg_cpu(int *NN,real *dmg, int *fail, real *fac);
void set_crack_cpu( real clength, int *NN, int *fail, int *NL, real *x, real *y);
void initialil_parameter(device_parameter &p, BaseModel_Parameter &bp);
#endif //PD_PARALLEL_BASE_FUNCTION_CUH
