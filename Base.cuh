//
// Created by wxm on 2021/12/2.
//

#ifndef PD_PARALLEL_BASE_CUH
#define PD_PARALLEL_BASE_CUH
#ifdef DOUBLE_PRECISION
typedef  double real;
#else
typedef float real;
#endif
#include "Error.cuh"
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sys/time.h>
using namespace std;

inline real square(real x) {return x*x;};
inline real cubic(real x) {return x*x*x;};

inline __device__ real gsquare(real x) {return x*x;};
inline __device__ real gcubic(real x) {return x*x*x;};
const int  FULL_MASK =  0xffffffff;
//二、三维下最大邻居数
const int MN_1D = 6;
const int MN_2D = 32;
const int MN_3D = 128;

const real pi = acos(-1.0);
const int  block_size = 256;


struct Dselect{
    Dselect( int device, int dim, int b_s, int mode) : device(device), Dim(dim), B_S(b_s), Mode(mode)
    {}
    int device;  //所用设备，cpu：1， gpu：2
    int Dim;    //记录模型维度，一维：1，二维：2，三维：3
    int B_S;    //求解方法,键型：1，态型：2
    int Mode;    //准静态问题：1， 瞬态问题：2

};


struct device_parameter{

    real pi;
    int N;
    int MN;
    int N_int;
    int nx;
    int ny;
    int nz;
    real dx;
    real emod;
    real dens;
    real pratio;
    real bc;
    real delta;
    real critical_s;
    real vol;
    real dt;
    int nt;
    real mass;
    real K;
    real G;
};
__constant__  extern device_parameter dp;
extern device_parameter p;

struct BaseModel_Parameter
{
    BaseModel_Parameter(const Dselect &d, real dt, real nt, real dx, real load=0.0, real thick=0.0,
                        real length=0.05, real height=0.05, real width=0.0,real emod=192.0e9, real pratio=1.0/3.0,
                        real dens=8000.0, real critict_s=0.04472)
            : dt(dt), nt(nt), dx(dx), load(load), thick(thick) ,length(length), height(height), width(width), emod(emod), pratio(pratio),
              dens(dens), critical_s(critict_s)
    {
        cal_parameter(d);
    }
    real load;     //荷载
    int nx;       //
    int ny;
    int nz;
    int N_int;  //积分点数
    int N;      //总质点数
    int MN;     //质点最大邻域点数

    real length;            //模型长
    real height;   //模型高
    real width;
    real dt;       //时间步长
    real nt;         //总时间步
    real vol;

    real dx;                    //质点间隔
    real radij;             //半质点间隔
    real bc;                //键常数

    real sedload;           //经典应变能
    real mass;               //质点质量
    real delta;               //近场范围
    real thick;                 //模型厚度
    real area;                  //质地面积
    real emod;                  //弹性模量
    real pratio;                //泊松比
    real dens = 7850.0;         //密度
    real K;                     //体积模量
    real G;                     //剪切模量
    real critical_s = 0.004472;      //临界键伸长率


    void cal_parameter(const Dselect &d)
    {
        delta = 3.015*dx;
        radij = dx/2.0;
        sedload = 0.5 * emod/(1.0-pratio*pratio) * 1.0e-6;
        nx = (length-dx/2.0)/dx + 1;
        if (d.B_S==1)
        {
            switch (d.Dim)
            {
                case 1:
                    MN = MN_1D;
                    area = dx*dx;
                    bc = 2.0*emod/(area* square(delta));
                    N = N_int = nx;
                    break;
                case 2:
                    MN = MN_2D;
                    ny = (height-dx/2.0)/dx + 1;
                    N = N_int = nx*ny;
                    vol = dx*dx*thick;
                    bc =  6.0 * emod / (pi*thick*cubic(delta)*(1.0 - pratio));
                    break;
                case 3:
                    MN = MN_3D;
                    ny = (height-dx/2.0)/dx + 1;
                    nz = (width-dx/2.0)/dx + 1;
                    N = N_int = nx*ny*nz;
                    vol = dx*dx*dx;
                    bc =  6.0 * emod / (pi*delta*cubic(delta)*(1.0 - 2.0*pratio));

                    break;
                default:
                    break;
            }

            mass = 0.25 * dt * dt * (pi*delta*delta*thick) * (bc) / dx*5.0;
        }
        else
        {
            G = 0.5*emod/(1+pratio);
            switch (d.Dim)
            {
                case 1:
                    MN = MN_1D;
                    N = N_int = nx;
                    area = dx*dx;
                    break;
                case 2:
                    MN = MN_2D;
                    ny = (height-dx/2.0)/dx + 1;
                    N = N_int = nx*ny;
                    thick = dx;
                    vol = dx*dx*thick;
                    K = 0.5*emod/(1-pratio);
                    break;
                case 3:
                    MN = MN_3D;
                    ny = (height-dx/2.0)/dx + 1;
                    nz = (width-dx/2.0)/dx + 1;
                    N = N_int = nx*ny*nz;
                    K = emod/3.0/(1-2.0*G);
                    vol = dx*dx*dx;
                    break;
                default:
                    break;

            }
        }
    }
};

struct BAtom{
    int *NN;
    real *x;
    real *y;
    real *z;
    real *disp_x;
    real *disp_y;
    real *disp_z;
    real *vel_x;
    real *vel_y;
    real *vel_z;

    real *acc_x;
    real *acc_y;
    real *acc_z;
    real *dmg;
    real *m;
    real *theta;

    real *bforce_x;
    real *bforce_y;
    real *bforce_z;
    real *pforce_x;
    real *pforce_y;
    real *pforce_z;
    real *velhalf_x;
    real *velhalf_y;
    real *velhalf_z;

    real *velhalfold_x;
    real *velhalfold_y;
    real *velhalfold_z;

    real *pforceold_x;
    real *pforceold_y;
    real *pforceold_z;
    real *cn_xy;
    real *disp_xy;

};

struct Cylinder_Parameter {

    real rad;
    real mass;
    real height;
    real v;
    real x;
    real y;
    real z;
};

struct Cylinder{
    real *acc;
    real *vel;
    real *disp;
    real *force;
};

extern __constant__ Cylinder_Parameter dcyp;
extern Cylinder_Parameter cyp;

struct BBond{
    int *NL;
    real *fac;
    int *fail;
    real *w; //储存键断裂信息
    real *idist;
    real *scr;
    real *e;

    real *nlength;
    real *bond_length;
};






#endif //PD_PARALLEL_BASE_CUH
