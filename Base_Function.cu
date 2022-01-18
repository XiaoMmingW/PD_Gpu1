//
// Created by wxm on 2021/12/2.
//

#include "Base_Function.cuh"

long double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (tp.tv_sec*1.0e6+(real)tp.tv_usec);
}

void initialil_parameter(device_parameter &p, BaseModel_Parameter &bp)
{
    p.pi = acos(-1.0);
    p.N = bp.N;
    p.MN = bp.MN;
    p.N_int = bp.N_int;
    p.nx = bp.nx;
    p.ny = bp.ny;
    p.nz = bp.nz;
    p.dens =  bp.dens;
    p.pratio = bp.pratio;
    p.bc = bp.bc;
    p.dx = bp.dx;
    p.delta = bp.delta;
    p.emod = bp.emod;
    p.critical_s = bp.critical_s;
    p.vol = bp.vol;
    p.dt = bp.dt;
    p.nt = bp.nt;
    p.mass = bp.mass;
    p.K = bp.K;
    p.G = bp.G;
}


__global__ void kernel_set_crack_2D(
        real clength, real loc_x, real loc_y,real theta, int *NN, int *fail, int *NL, real *x, real *y)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    unsigned int cnode = 0;

    if (i<dp.N_int)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];

            if(fabs(theta-dp.pi/2.0)<0.0001)
            {
                if(fabs(y[cnode]-loc_y) < clength*sin(theta)/2.0 && fabs(y[i]-loc_y) < clength*sin(theta)/2.0)
                {
                    if((x[cnode]-loc_x)*(x[i]-loc_x) < 0.0 )
                    {
                        fail[idx] = 0;
                    }
                }
            }
            else
            {
                real dist_cnode = gsquare(y[cnode]-loc_y)+ gsquare(x[cnode]-loc_x);
                real dist_i = gsquare(y[i]-loc_y)+ gsquare(x[i]-loc_x);
                real vertical_dcnode = gsquare((tan(theta)*x[cnode]-y[cnode]-loc_x*tan(theta)+loc_y) / sqrt(1+ tan(theta)*tan(theta)));
                real vertical_di = gsquare((tan(theta)*x[i]-y[i]-loc_x*tan(theta)+loc_y) / sqrt(1+ tan(theta)*tan(theta)));
                real length_cnode = sqrt(dist_cnode - vertical_dcnode);
                real length_i = sqrt(dist_i - vertical_di);
                if(length_cnode<clength/2.0 && length_i<clength/2.0)
                {
                    real bcnode= tan(theta)*x[cnode]-loc_x*tan(theta)+loc_y - y[cnode];
                    real bi= tan(theta)*x[i]-loc_x*tan(theta)+loc_y - y[i];
                    if(bcnode * bi<=0)
                        fail[idx] = 0;
                }
            }
        }
    }
}

void set_crack_2D_cpu(real clength, real loc_x, real loc_y,real theta, int *NN, int *fail, int *NL, real *x, real *y)
{

    unsigned int cnode = 0;
    unsigned int idx = 0;

    for (int i=0; i<p.N_int; i++)
    {
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*p.MN + j;
            cnode = NL[idx];
            if(fabs(theta-p.pi/2.0)<0.0001)
            {
                if(fabs(y[cnode]-loc_y) < clength*sin(theta)/2.0 && fabs(y[i]-loc_y) < clength*sin(theta)/2.0)
                {
                    if((x[cnode]-loc_x)*(x[i]-loc_x) < 0.0 )
                    {
                        fail[idx] = 0;
                    }
                }
            }
            else
            {
                real dist_cnode = square(y[cnode]-loc_y)+ square(x[cnode]-loc_x);
                real dist_i = square(y[i]-loc_y)+ square(x[i]-loc_x);
                real vertical_dcnode = square((tan(theta)*x[cnode]-y[cnode]-loc_x*tan(theta)+loc_y) / sqrt(1+ tan(theta)*tan(theta)));
                real vertical_di = square((tan(theta)*x[i]-y[i]-loc_x*tan(theta)+loc_y) / sqrt(1+ tan(theta)*tan(theta)));
                real length_cnode = sqrt(dist_cnode - vertical_dcnode);
                real length_i = sqrt(dist_i - vertical_di);
                if(length_cnode<clength/2.0 && length_i<clength/2.0)
                {
                    real bcnode= tan(theta)*x[cnode]-loc_x*tan(theta)+loc_y - y[cnode];
                    real bi= tan(theta)*x[i]-loc_x*tan(theta)+loc_y - y[i];
                    if(bcnode * bi<=0)
                        fail[idx] = 0;
                }
            }
        }
    }
}

void set_crack_cpu( real clength, int *NN, int *fail, int *NL, real *x, real *y)
{
    unsigned int cnode = 0;
    for (int i=0; i<p.N_int; i++)
    {
        for (int j=0; j<NN[i]; j++)
        {
            cnode = NL[i*p.MN+j];
            if (y[cnode]>0.0 && y[i]<0.0)
            {
                if(fabs(x[i])-clength/2.0<1.0e-10)
                    fail[i*p.MN+j] = 0;
                else if (fabs(x[cnode])-clength/2.0<1.0e-10)
                    fail[i*p.MN+j] = 0;
            } else if (y[i]>0.0 && y[cnode]<0.0){
                if(fabs(x[i])-clength/2.0<1.0e-10)
                    fail[i*p.MN+j] = 0;
                else if (fabs(x[cnode])-clength/2.0<1.0e-10)
                    fail[i*p.MN+j] = 0;
            }
        }
    }
}

__global__ void kernel_cal_dmg_2D(int *NN,real *dmg, int *fail, real *fac)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;

    if (i<dp.N_int)
    {
        real dmgpar1 = 0.0;
        real dmgpar2 = 0.0;

        if (j<NN[i])
        {

            dmgpar1 = fail[idx]*dp.vol*fac[idx];
            dmgpar2 = dp.vol * fac[idx];
        }

        for (int offset=16; offset>0; offset>>=1)
        {
            dmgpar1 += __shfl_down_sync(FULL_MASK, dmgpar1, offset);
            dmgpar2 += __shfl_down_sync(FULL_MASK, dmgpar2, offset);
        }
        if (j==0) dmg[i] = 1.0 - dmgpar1/dmgpar2;
    }
}

__global__ void kernel_cal_dmg_3D(int *NN,real *dmg, int *fail, real *fac)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    extern __shared__ real dy_dmg[];
    real *dmg1 = dy_dmg;
    real *dmg2 = (real*)&dmg1[block_size];
    if (i<dp.N_int)
    {
        real dmgpar1 = 0.0;
        real dmgpar2 = 0.0;


        if (j<NN[i])
        {
            dmgpar1 = fail[idx]*dp.vol*fac[idx];
            dmgpar2 = dp.vol * fac[idx];
        }

        for (int offset=16; offset>0; offset>>=1)
        {
            dmgpar1 += __shfl_down_sync(FULL_MASK, dmgpar1, offset);
            dmgpar2 += __shfl_down_sync(FULL_MASK, dmgpar2, offset);
        }

        if(tid%32==0)
        {
            dmg1[tid] = dmgpar1;
            dmg2[tid] = dmgpar2;

        }
        __syncthreads();
        if (j==0)
        {
            dmg[i] = 1.0-(dmg1[tid]+dmg1[tid+32]+dmg1[tid+64]+dmg1[tid+96])/(dmg2[tid]+dmg2[tid+32]+dmg2[tid+64]+dmg2[tid+96]);

        }

    }
}


void cal_dmg_cpu(int *NN,real *dmg, int *fail, real *fac)
{

    unsigned int idx =0;
    real dmgpar1 = 0.0;
    real dmgpar2 = 0.0;
    for (int i=0; i<p.N_int; i++)
    {
        dmgpar1 = 0.0;
        dmgpar2 = 0.0;
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*p.MN + j;
            dmgpar1 += fail[idx]*p.vol*fac[idx];
            dmgpar2 += p.vol * fac[idx];
        }
        dmg[i] = 1.0 - dmgpar1/dmgpar2;
    }
}



