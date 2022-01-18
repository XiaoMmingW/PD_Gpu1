//
// Created by wxm on 2021/12/4.
//

#include "Force_Bond.cuh"
__global__  void kernel_bond_force_2D(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/dp.MN;
    int j = idx%dp.MN;
    int cnode = 0;
    real nlength = 0.0;
    real s = 0.0;
    real force_x = 0.0;
    real force_y = 0.0;

    if (i<dp.N)
    {
        if(fail[idx]==1)
        {
            if(j<NN[i])
            {
                cnode = NL[idx];

                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
                s = (nlength - idist[idx])/idist[idx];
                force_x =  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(x[cnode] + disp_x[cnode] - x[i]- disp_x[i])/nlength;
                force_y =  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(y[cnode] + disp_y[cnode] - y[i]- disp_y[i])/nlength;
                if (s>dp.critical_s) fail[idx] = 0;
            }
        }
        //__syncwarp();

        for (int offset = 16; offset>0; offset>>=1)
        {
            force_x += __shfl_down_sync(FULL_MASK, force_x, offset);
            force_y += __shfl_down_sync(FULL_MASK, force_y, offset);
        }
        // __syncwarp();
        if (j==0)
        {
            pforce_x[i] = force_x;
            pforce_y[i] = force_y;
        }

    }
}

__global__  void kernel_particle_force_2D(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx = 0;
    int cnode = 0;
    real nlength = 0.0;
    real s = 0.0;
    real force_x = 0.0;
    real force_y = 0.0;

    if (i<dp.N)
    {
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*dp.MN + j;
            if(fail[idx]==1)
            {
                cnode = NL[idx];
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
                s = (nlength - idist[idx])/idist[idx];
                force_x +=  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(x[cnode] + disp_x[cnode] - x[i]- disp_x[i])/nlength;
                force_y +=  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(y[cnode] + disp_y[cnode] - y[i]- disp_y[i])/nlength;
                if (s>dp.critical_s) fail[idx] = 0;
            }
        }

        pforce_x[i] = force_x;
        pforce_y[i] = force_y;

    }
}

__global__  void kernel_bond_force_3D(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, real *z, real *disp_z,
        real *pforce_z, int *NN, int *NL, real *idist, real *scr, int *fail, real *fac
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    extern  __shared__ real force[];
    real *fx = force;
    real *fy = (real *)&fx[block_size];
    real *fz = (real *)&fy[block_size];
    real force_x = 0.0;
    real force_y = 0.0;
    real force_z = 0.0;
    if (i<dp.N)
    {
        if(fail[idx]==1)
        {
            if(j<NN[i])
            {
                unsigned int cnode = NL[idx];
                real nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                                    gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i])+
                                    gsquare(z[cnode]+disp_z[cnode]-z[i]-disp_z[i]));
                real s = (nlength - idist[idx])/idist[idx];
                force_x =  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])/nlength;
                force_y =  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(y[cnode]+disp_y[cnode]-y[i]-disp_y[i])/nlength;
                force_z =  dp.vol * fac[idx] * dp.bc*scr[idx]*s*(z[cnode]+disp_z[cnode]-z[i]-disp_z[i])/nlength;
                if (s>dp.critical_s && y[i]>=-0.04&&y[cnode]>=-0.04) fail[idx] = 0;
            }
        }
    }
    for (int offset = 16; offset>0; offset>>=1)
    {
        force_x += __shfl_down_sync(FULL_MASK, force_x, offset);
        force_y += __shfl_down_sync(FULL_MASK, force_y, offset);
        force_z += __shfl_down_sync(FULL_MASK, force_z, offset);
    }
    if(tid%32==0)
    {
        fx[tid] = force_x;
        fy[tid] = force_y;
        fz[tid] = force_z;
    }
    __syncthreads();
    if (j==0)
    {
        pforce_x[i] = fx[tid]+fx[tid+32]+fx[tid+64]+fx[tid+96];
        pforce_y[i] = fy[tid]+fy[tid+32]+fy[tid+64]+fy[tid+96];
        pforce_z[i] = fz[tid]+fz[tid+32]+fz[tid+64]+fz[tid+96];
        //if(i==0) printf("pforce_y %e \n", pforce_y[i]);
    }
}

void bond_force_2D_gpu(BAtom &ba, BBond &bb)
{
    int grid_size = (p.N*p.MN-1)/block_size + 1;
    kernel_bond_force_2D<<<grid_size,block_size>>>(
            ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y, ba.NN, bb.NL,
            bb.idist, bb.scr, bb.fail, bb.fac);
}

void bond_force_3D_gpu(BAtom &ba, BBond &bb)
{
    int grid_size = (p.N*p.MN-1)/block_size + 1;
    kernel_bond_force_3D<<<grid_size,block_size, block_size>>>(
            ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y, ba.z, ba.disp_z, ba.pforce_z,ba.NN, bb.NL,
            bb.idist, bb.scr, bb.fail, bb.fac);
}

void bond_force_cpu(
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, int *NN, int *NL,
        real *idist, real *scr, int *fail, real *fac, real *z, real *disp_z, real *pforce_z
)
{

    int cnode = 0;
    real nlength = 0.0;
    real s = 0.0;
    int idx = 0;
    for (int i=0; i<p.N; i++)
    {
        pforce_x[i] = 0.0;
        pforce_y[i] = 0.0;
        if(z!=NULL) pforce_z[i] = 0.0;
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*p.MN + j;
            if (fail[idx]==1)
            {
                cnode = NL[idx];
                if (z==NULL)
                {
                    nlength = sqrt(square(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                                   square(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
                } else {
                    nlength = sqrt(square(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                                   square(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]) +
                                   square(z[cnode]+disp_z[cnode]-z[i]-disp_z[i]));
                }
                s = (nlength - idist[idx])/idist[idx];
                pforce_x[i] +=  p.vol * fac[idx] * p.bc*scr[idx]*s*(x[cnode] + disp_x[cnode] - x[i]- disp_x[i])/nlength;
                pforce_y[i] +=  p.vol * fac[idx] * p.bc*scr[idx]*s*(y[cnode] + disp_y[cnode] - y[i]- disp_y[i])/nlength;
                if(z!=NULL)
                {  pforce_z[i] +=  p.vol * fac[idx] * p.bc*scr[idx]*s*(z[cnode] + disp_z[cnode] - z[i]- disp_z[i])/nlength;
                    if (s>p.critical_s && y[i]>=-0.04&&y[cnode]>=-0.04) fail[idx] = 0;
                } else{
                    if (s>p.critical_s) fail[idx] = 0;

                }
            }
        }
    }
}


