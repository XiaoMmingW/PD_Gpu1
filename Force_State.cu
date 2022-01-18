//
// Created by wxm on 2021/12/4.
//

#include "Force_State.cuh"


__global__ void kernel_initial_weight_2D(
        real *m, int *NN, real *w, real *idist, real *fac)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    real temp_m = 0.0;
    if (i<dp.N)
    {
        if (j<NN[i])
        {
            w[idx] = exp(-gsquare(idist[idx]) / gsquare(dp.delta));
            temp_m = w[idx] * gsquare(idist[idx]) * dp.vol * fac[idx];
        }
        for (int offset=16; offset>0; offset>>=1)
            temp_m += __shfl_down_sync(FULL_MASK, temp_m, offset);
        if (j==0)
            m[i] = temp_m;
    }
}

__global__ void kernel_initial_weight_3D(
        real *m, int *NN, real *w, real *idist, real *fac)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    real temp_m = 0.0;
    extern __shared__ real mm[];
    if (i<dp.N)
    {
        if (j<NN[i])
        {
            w[idx] = exp(-gsquare(idist[idx]) / gsquare(dp.delta));
            temp_m = w[idx] * gsquare(idist[idx]) * dp.vol * fac[idx];
        }
        for (int offset=16; offset>0; offset>>=1)
            temp_m += __shfl_down_sync(FULL_MASK, temp_m, offset);
        if(tid%32==0)
            mm[tid] = temp_m;
        __syncthreads();
        if (j==0)
            m[i] = mm[tid]+mm[tid+32]+mm[tid+64]+mm[tid+96];
        //if(i==0) printf("NN %d %e %e %e %e\n", NN[i],mm[0], mm[0+32], mm[0+64], mm[0+96]);
    }
}

__global__ void kernel_cal_theta_2D(
        int *NN, real *m, real *theta, int *NL, real *w, int *fail, real *e, real *idist, real *fac,
        real *x, real *disp_x, real *y, real *disp_y)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    unsigned int cnode = 0;
    real temp_theta = 0.0;
    //if (fail[idx]==1)
    {
        if (i < dp.N)
        {
            if (j < NN[i])
            {
                cnode = NL[idx];
                e[idx] = sqrt(gsquare(x[cnode] - x[i] + disp_x[cnode] - disp_x[i]) +
                              gsquare(y[cnode] - y[i] + disp_y[cnode]-  disp_y[i])) - idist[idx];
                temp_theta = 2.0*(2.0*dp.pratio-1.0)/(dp.pratio-1.0)/m[i]*w[idx]*idist[idx]*e[idx]*fac[idx]*dp.vol;
                //if(i==0) printf("temp_theta %e idist %e m %e\n", e[idx], idist[idx], m[i]);
            }
            __syncwarp();
            for (int offset = 16; offset>0; offset>>=1)
                temp_theta += __shfl_down_sync(FULL_MASK, temp_theta, offset);
            if (j==0)
                theta[i] = temp_theta;
            //if(idx==0) printf("theta %e\n", theta[i]);
        }
    }
}


__global__  void kernel_state_force_2D(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist,
        real *fac, real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    unsigned int cnode = 0;
    real nlength = 0.0;
    real force_x = 0.0;
    real force_y = 0.0;
    if (i<dp.N_int)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            if (fail[idx] == 1)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
                real s = (nlength - idist[idx])/idist[idx];;

                real ti = (2.0*dp.pratio-1.0)/(dp.pratio-1.0)*(2.0*dp.K-32.0*dp.G/9.0)*theta[i]*w[idx]*idist[idx]/m[i]
                          + 8.0*dp.G/m[i]*w[idx]*e[idx];

                real tj = (2.0*dp.pratio-1.0)/(dp.pratio-1.0)*(2.0*dp.K-32.0*dp.G/9.0)*theta[cnode]*w[idx]*idist[idx]/m[cnode]
                          + 8.0*dp.G/m[cnode]*w[idx]*e[idx];
                force_x = (ti*dp.vol+tj*dp.vol)*fac[idx]*(x[cnode] + disp_x[cnode] - x[i]- disp_x[i]) / nlength;
                force_y = (ti*dp.vol+tj*dp.vol)*fac[idx]*(y[cnode] + disp_y[cnode] - y[i]- disp_y[i]) / nlength;
                if (s>dp.critical_s) fail[idx] = 0;
            }
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_x += __shfl_down_sync(FULL_MASK, force_x, offset);
            force_y += __shfl_down_sync(FULL_MASK, force_y, offset);
        }

        if (j==0)
        {
            pforce_x[i] = force_x;
            pforce_y[i] = force_y;
        }
    }
}

__global__ void kernel_cal_theta_2D_particle(
        int *NN, real *m, real *theta, int *NL, real *w, int *fail, real *e, real *idist, real *fac,
        real *x, real *disp_x, real *y, real *disp_y)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx = 0;
    unsigned int cnode = 0;
    real temp_theta = 0.0;
    //if (fail[idx]==1)
    {
        if (i < dp.N)
        {
            for (int j=0; j<NN[i]; j++)
            {
                idx = i*dp.MN+j;
                cnode = NL[idx];
                e[idx] = sqrt(gsquare(x[cnode] - x[i] + disp_x[cnode] - disp_x[i]) +
                              gsquare(y[cnode] - y[i] + disp_y[cnode]-  disp_y[i])) - idist[idx];
                temp_theta += 2.0*(2.0*dp.pratio-1.0)/(dp.pratio-1.0)/m[i]*w[idx]*idist[idx]*e[idx]*fac[idx]*dp.vol;
                //if(i==0) printf("temp_theta %e idist %e m %e\n", e[idx], idist[idx], m[i]);
            }
            theta[i] = temp_theta;
        }
    }
}


__global__  void kernel_state_force_2D_particle(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist,
        real *fac, real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y
)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx = 0;
    unsigned int cnode = 0;
    real nlength = 0.0;
    real force_x = 0.0;
    real force_y = 0.0;
    if (i<dp.N_int)
    {
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*dp.MN+j;
            cnode = NL[idx];
            if (fail[idx] == 1)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
                real s = (nlength - idist[idx])/idist[idx];

                real ti = (2.0*dp.pratio-1.0)/(dp.pratio-1.0)*(2.0*dp.K-32.0*dp.G/9.0)*theta[i]*w[idx]*idist[idx]/m[i]
                          + 8.0*dp.G/m[i]*w[idx]*e[idx];

                real tj = (2.0*dp.pratio-1.0)/(dp.pratio-1.0)*(2.0*dp.K-32.0*dp.G/9.0)*theta[cnode]*w[idx]*idist[idx]/m[cnode]
                          + 8.0*dp.G/m[cnode]*w[idx]*e[idx];
                force_x += (ti*dp.vol+tj*dp.vol)*fac[idx]*(x[cnode] + disp_x[cnode] - x[i]- disp_x[i]) / nlength;
                force_y += (ti*dp.vol+tj*dp.vol)*fac[idx]*(y[cnode] + disp_y[cnode] - y[i]- disp_y[i]) / nlength;
                if (s>dp.critical_s) fail[idx] = 0;
            }
        }
        pforce_x[i] = force_x;
        pforce_y[i] = force_y;
    }
}



__global__ void kernel_cal_theta_3D(
        int *NN, real *m, real *theta, int *NL, real *w, int *fail, real *e, real *idist, real *fac,
        real *x, real *disp_x, real *y, real *disp_y, real *z, real *disp_z)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    unsigned int cnode = 0;
    real temp_theta = 0.0;
    extern __shared__ real ff[];
    if (fail[idx]==1)
    {
        if (i < dp.N)
        {
            if (j < NN[i])
            {
                cnode = NL[idx];
                e[idx] = sqrt(gsquare(x[cnode] - x[i] + disp_x[cnode] - disp_x[i]) +
                              gsquare(y[cnode] - y[i] + disp_y[cnode]-  disp_y[i]) +
                              gsquare(z[cnode] - z[i] + disp_z[cnode]-  disp_z[i])) - idist[idx];
                temp_theta = 3.0/m[i]*w[idx]*idist[idx]*e[idx]*fac[idx]*dp.vol;
                //if(i==0) printf("temp_theta %e idist %e m %e\n", e[idx], idist[idx], m[i]);
            }

            for (int offset = 16; offset>0; offset>>=1)
                temp_theta += __shfl_down_sync(FULL_MASK, temp_theta, offset);
            if(tid%32==0)
            {
                ff[tid] = temp_theta;
            }
            __syncthreads();
            if (j==0)
                theta[i] = ff[tid]+ff[tid+32]+ff[tid+64]+ff[tid+96];
        }
    }
}


__global__  void kernel_state_force_3D(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist, real *fac,
        real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y, real *z, real *disp_z, real *pforce_z
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    unsigned int cnode = 0;
    real nlength = 0.0;
    real force_x = 0.0;
    real force_y = 0.0;
    real force_z = 0.0;
    extern  __shared__ real force[];
    real *fx = force;
    real *fy = (real *)&fx[block_size];
    real *fz = (real *)&fy[block_size];
//__shared__ real fx[256];
//__shared__ real fy[256];
//__shared__ real fz[256];
    if (i<dp.N_int)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            if (fail[idx] == 1)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i])+
                               gsquare(z[cnode] - z[i] + disp_z[cnode]-  disp_z[i]));
                real s = (nlength - idist[idx])/idist[idx];;
                real ti = 3.0*dp.K*theta[i]*w[idx]*idist[idx]/m[i] + 15.0*dp.G/m[i]*w[idx]*(e[idx]-theta[i]*idist[idx]/3.0);
                real tj = 3.0*dp.K*theta[cnode]*w[idx]*idist[idx]/m[cnode] +
                          15.0*dp.G/m[cnode]*w[idx]*(e[idx]-theta[cnode]*idist[idx]/3.0);
                force_x = (ti*dp.vol+tj*dp.vol)*fac[idx]*(x[cnode] + disp_x[cnode] - x[i]- disp_x[i]) / nlength;
                force_y = (ti*dp.vol+tj*dp.vol)*fac[idx]*(y[cnode] + disp_y[cnode] - y[i]- disp_y[i]) / nlength;
                force_z = (ti*dp.vol+tj*dp.vol)*fac[idx]*(z[cnode] + disp_z[cnode] - z[i]- disp_z[i]) / nlength;
                if (s>dp.critical_s && y[i]>=-0.04&&y[cnode]>=-0.04) fail[idx] = 0;
            }
        }
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_x += __shfl_down_sync(FULL_MASK, force_x, offset);
            force_y += __shfl_down_sync(FULL_MASK, force_y, offset);
            force_z += __shfl_down_sync(FULL_MASK, force_z, offset);
        }
        __syncwarp();
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

            //if(i==0) printf("theta %e force %e \n",theta[i],pforce_x[i]);
        }
    }
}


void initial_weight_cpu(real *m, int *NN,  real *w, real *idist, real *fac)
{
    for(int i=0; i<p.N; i++)
    {
        m[i] = 0.0;
        for(int j=0; j<NN[i]; j++)
        {
            int idx = i*p.MN + j;
            w[idx] = exp(-square(idist[idx]) / square(p.delta));
            m[i] += w[idx] * square(idist[idx]) * p.vol * fac[idx];
        }
    }
}

void state_force_2D_cpu(
        int *NN, real *m, real *theta,  int *fail, int *NL, real *e, real *w, real *idist,
        real *fac, real *x, real *disp_x, real *pforce_x, real *y, real *disp_y, real *pforce_y
)
{

    unsigned int cnode = 0;
    real nlength = 0.0;
    unsigned int idx = 0;
    real s;
    real nx;
    real ny;
    real ti;
    real tj;


    for (int i=0; i<p.N; i++)
    {
        theta[i] = 0.0;
        for (int j=0; j<NN[i]; j++)
        {
            int idx = i*p.MN+j;
            if(fail[idx]==1)
            {
                idx = i*p.MN +j;
                cnode = NL[idx];
                e[idx] = sqrt(square(x[cnode] - x[i] + disp_x[cnode] - disp_x[i]) +
                              square(y[cnode] - y[i] + disp_y[cnode]-  disp_y[i])) - idist[idx];
                theta[i] += 2.0*(2.0*p.pratio-1.0)/(p.pratio-1.0)/m[i]*w[idx]*idist[idx]*e[idx]*fac[idx]*p.vol;
                //if(i==0) cout<<"theta "<<theta[i]<<" "<<idist[idx]<<endl;
            }

        }

    }

    for (int i=0; i<p.N_int; i++)
    {
        pforce_x[i] = 0.0;
        pforce_y[i] = 0.0;
        for (int j=0; j<NN[i]; j++)
        {

            idx = i*p.MN +j;
            //if(i==0) cout<<"fail "<<fail[idx]<<" "<<w[idx]<<endl;
            if(fail[idx]==1)
            {
                cnode = NL[idx];

                nlength = sqrt(square(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               square(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
                s = (nlength - idist[idx])/idist[idx];
                nx = (x[cnode] + disp_x[cnode] - x[i]- disp_x[i]) / nlength;
                ny = (y[cnode] + disp_y[cnode] - y[i]- disp_y[i]) / nlength;

                ti = (2.0*p.pratio-1.0)/(p.pratio-1.0)*(2.0*p.K-32.0*p.G/9.0)*theta[i]*w[idx]*idist[idx]/m[i]
                     + 8.0*p.G/m[i]*w[idx]*e[idx];

                tj = (2.0*p.pratio-1.0)/(p.pratio-1.0)*(2.0*p.K-32.0*p.G/9.0)*theta[cnode]*w[idx]*idist[idx]/m[cnode]
                     + 8.0*p.G/m[cnode]*w[idx]*e[idx];
                pforce_x[i] += (ti*p.vol+tj*p.vol)*nx*fac[idx];
                pforce_y[i] += (ti*p.vol+tj*p.vol)*ny*fac[idx];

                if (s>p.critical_s) fail[idx] = 0;
                //if(i==0) cout<<"fail "<<fail[idx]<<" s "<<idist[idx]<<" nlength "<<nlength<<endl;
                // if(i==0) cout<<"force "<<pforce_y[i]<<" "<<m[i]<<" "<<m[cnode]<<endl;
            }
        }

    }
}

