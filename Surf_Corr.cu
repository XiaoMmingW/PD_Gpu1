//
// Created by wxm on 2021/12/2.
//

#include "Surf_Corr.cuh"

__global__ void kernel_vol_Corr
        (int *NN, int *NL, real *x, real *idist, real *fac, real *y, real *z)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;

    int i = idx/dp.MN;
    int j = idx%dp.MN;

    if (i<dp.N)
    {
        if (j<NN[i])
        {
            int cnode = NL[idx];
            if (y==NULL && z==NULL)
                idist[idx] = sqrt(gsquare(x[cnode]-x[i]));
            if (y!=NULL && z==NULL)
                idist[idx] = sqrt(gsquare(x[cnode]-x[i]) + gsquare(y[cnode]-y[i]));
            if (y!=NULL && z!=NULL)
                idist[idx] = sqrt(gsquare(x[cnode]-x[i]) + gsquare(y[cnode]-y[i]) + gsquare(z[cnode]-z[i]));

            if (idist[idx] <= dp.delta-dp.dx/2.0)
                fac[idx] = 1.0;
            else if (idist[idx] <= dp.delta+dp.dx/2.0)
                fac[idx] = (dp.delta+dp.dx/2.0-idist[idx]) / dp.dx;
            else
                fac[idx] = 0.0;
        }
    }
}


static __global__ void kernel_Disp
        (real *coord, real *disp_cal, real *disp_initial_1=NULL, real *disp_initial_2=NULL)
{
    int i =  blockIdx.x*blockDim.x + threadIdx.x;
    if (i<dp.N)
    {
        disp_cal[i] = 0.001 * coord[i];
        if(disp_initial_1!=NULL)
            disp_initial_1[i] = 0.0;
        if(disp_initial_2!=NULL)
            disp_initial_2[i] = 0.0;
    }
}


static __global__ void kernel_surface_F
        (real sedload_Cal, int *NN, int *NL, real *x, real *disp_x,real *fncst_Cal,  real *idist,
         real *fac,  real *y=NULL, real *disp_y=NULL,  real *z=NULL, real *disp_z=NULL)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx/dp.MN;
    int j = idx%dp.MN;
    int  cnode = 0;
    real nlength =0.0;
    real stendens = 0.0;

    if(i<dp.N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            if (y==NULL && z==NULL)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i]));
            }
            if (y!=NULL && z==NULL)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
            }
            if (y!=NULL && z!=NULL)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i])+
                               gsquare(z[cnode]+disp_z[cnode]-z[i]-disp_z[i]));
            }
            stendens = 0.25*dp.bc*(nlength-idist[idx])*(nlength-idist[idx])/idist[idx]*dp.vol*fac[idx];

        }
        __syncwarp();

        for (int offset = dp.MN>>1; offset>0; offset>>=1)
        {
            stendens += __shfl_down_sync(FULL_MASK, stendens, offset);

        }
        if (j==0)
        {
            fncst_Cal[i] = sedload_Cal / stendens;
        }

    }
}

static __global__ void kernel_surface_F_3D
(real sedload_Cal, int *NN, int *NL, real *x, real *disp_x,real *fncst_Cal,  real *idist,
 real *fac,  real *y=NULL, real *disp_y=NULL,  real *z=NULL, real *disp_z=NULL)
 {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx/dp.MN;
    int j = idx%dp.MN;
    int  cnode = 0;
    real nlength =0.0;
    real stendens = 0.0;
    extern __shared__ real temp[];
    unsigned int tid = threadIdx.x;
    if(i<dp.N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            if (y==NULL && z==NULL)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i]));
            }
            if (y!=NULL && z==NULL)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                        gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
            }
            if (y!=NULL && z!=NULL)
            {
                nlength = sqrt(gsquare(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                        gsquare(y[cnode]+disp_y[cnode]-y[i]-disp_y[i])+
                        gsquare(z[cnode]+disp_z[cnode]-z[i]-disp_z[i]));
            }
            stendens = 0.25*dp.bc*(nlength-idist[idx])*(nlength-idist[idx])/idist[idx]*dp.vol*fac[idx];

        }
        //__syncwarp();

        for (int offset = 16; offset>0; offset>>=1)
        {
            stendens += __shfl_down_sync(FULL_MASK, stendens, offset);
        }
        if(tid%32==0)
        {

            temp[tid] = stendens;


        }
        __syncthreads();
        if (j==0)
        {
            fncst_Cal[i] = sedload_Cal/(temp[tid]+temp[tid+32]+temp[tid+64]+temp[tid+96]);
            //if(i==0) printf("fnc %e \n",fncst_Cal[i]);
        }

    }
 }

static __global__ void kernel_cal_surf_coff_F
        (int *NN, int *NL, real *scr, real *x, real *fncst_x,
         real *y=NULL, real *fncst_y=NULL,  real *z=NULL, real *fncst_z=NULL, real *idist=NULL)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/dp.MN;
    int j = idx%dp.MN;
    int  cnode = 0;

    if (i<dp.N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];

            if(y == NULL)
            {
                scr[idx] = (fncst_x[i] + fncst_x[cnode]) / 2.0;

            }
            if (y != NULL && z==NULL)
            {
                real theta = 0.0;
                real scx = 0.0;
                real scy = 0.0;
                if (fabs(y[cnode] - y[i]) <= 1.0e-10)
                    theta = 0.0;
                else if (fabs(x[cnode] - x[i]) <= 1.0e-10)
                    theta = 90.0*dp.pi/180.0;
                else
                    theta = atan(fabs(y[cnode] - y[i])/fabs(x[cnode] - x[i]));
                scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
                scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;

                scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy)));
            }
            if (y != NULL && z!=NULL)
            {
                real theta = 0.0;
                real scx = 0.0;
                real scy = 0.0;
                real scz = 0.0;
                if(fabs(z[cnode]-z[i])<1.0e-10)
                {
                    if(fabs(y[cnode]-y[i])<1.0e-10)
                        theta = 0.0;
                    else if (fabs(x[cnode]-x[i])<1.0e-10)
                        theta = 90.0*dp.pi/180.0;
                    else
                        theta = atan(fabs(y[cnode] - y[i])/fabs(x[cnode] - x[i]));
                    real phi = 90.0*dp.pi/180.0;
                    scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
                    scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
                    scz = (fncst_z[i] + fncst_z[cnode]) / 2.0;
                    scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy) +
                                         cos(phi)*cos(phi)/(scz*scz)));
                }
                else if (fabs(x[cnode]-x[i])<1.0e-10 && fabs(y[cnode]-y[i])<1.0e-10)
                    scr[idx] = (fncst_z[i] + fncst_z[cnode]) / 2.0;
                else
                {
                    theta = atan(fabs(y[cnode] - y[i])/fabs(x[cnode] - x[i]));
                    real phi = acos(fabs(z[cnode]-z[i])/idist[idx]);
                    scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
                    scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
                    scz = (fncst_z[i] + fncst_z[cnode]) / 2.0;
                    scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy) +
                                         cos(phi)*cos(phi)/(scz*scz)));
                }
            }
        }
    }
}


void surface_correct_gpu(Dselect &d, BAtom &ba, BBond &bb)
{
    int grid_size = (p.N*p.MN-1)/block_size + 1;
    real sedload_Cal = 0.5*p.emod/(1.0-p.pratio*p.pratio) * 1.0e-6;
    if(d.Dim==1)
    {
        kernel_vol_Corr<<<grid_size, block_size>>>(ba.NN, bb.NL, ba.x, bb.idist, bb.fac);
        real *fncst_x;
        CHECK(cudaMalloc((void**)&fncst_x,p.N*sizeof(real)));

        kernel_Disp<<<grid_size, block_size>>>(ba.x, ba.disp_x);
        kernel_surface_F<<<grid_size, block_size>>>(
                sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_x,  bb.idist, bb.fac);
        kernel_cal_surf_coff_F<<<grid_size, block_size>>>(ba.NN, bb.NL, bb.scr, ba.x, fncst_x);
        CHECK(cudaFree(fncst_x));
    } else if(d.Dim==2)
    {
        kernel_vol_Corr<<<grid_size, block_size>>>(ba.NN, bb.NL, ba.x, bb.idist, bb.fac, ba.y);

        real *fncst_x;
        CHECK(cudaMalloc((void**)&fncst_x,p.N*sizeof(real)));
        real *fncst_y;
        CHECK(cudaMalloc((void**)&fncst_y,p.N*sizeof(real)));



        kernel_Disp<<<grid_size, block_size>>>(ba.x, ba.disp_x, ba.disp_y);

        kernel_surface_F<<<grid_size, block_size>>>(
                sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_x,  bb.idist, bb.fac, ba.y, ba.disp_y);

        kernel_Disp<<<grid_size, block_size>>>(ba.y, ba.disp_y, ba.disp_x);

        kernel_surface_F<<<grid_size, block_size>>>(
                sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_y,  bb.idist, bb.fac, ba.y, ba.disp_y);

        kernel_cal_surf_coff_F<<<grid_size, block_size>>>
                (ba.NN, bb.NL, bb.scr, ba.x, fncst_x, ba.y, fncst_y);

        CHECK(cudaFree(fncst_x));
        CHECK(cudaFree(fncst_y));
    } else
    {
        const int smem = block_size*sizeof(real);
        kernel_vol_Corr<<<grid_size, block_size>>>(ba.NN, bb.NL, ba.x, bb.idist, bb.fac, ba.y, ba.z);
        sedload_Cal =0.6*p.emod*1.0e-6;
        //cout<<"sed %e"<<sedload_Cal<<endl;
        real *fncst_x;
        CHECK(cudaMalloc((void**)&fncst_x,p.N*sizeof(real)));
        real *fncst_y;
        CHECK(cudaMalloc((void**)&fncst_y,p.N*sizeof(real)));
        real *fncst_z;
        CHECK(cudaMalloc((void**)&fncst_z,p.N*sizeof(real)));

        kernel_Disp<<<grid_size, block_size>>>(ba.x, ba.disp_x, ba.disp_y, ba.disp_z);

        kernel_surface_F_3D<<<grid_size, block_size,smem>>>(
                sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_x,  bb.idist, bb.fac, ba.y, ba.disp_y, ba.z, ba.disp_z);

        kernel_Disp<<<grid_size, block_size>>>(ba.y, ba.disp_y, ba.disp_x, ba.disp_z);

        kernel_surface_F_3D<<<grid_size, block_size, smem>>>(
                sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_y,  bb.idist, bb.fac, ba.y, ba.disp_y, ba.z, ba.disp_z);
        kernel_Disp<<<grid_size, block_size>>>(ba.z, ba.disp_z, ba.disp_x, ba.disp_y);

        kernel_surface_F_3D<<<grid_size, block_size, smem>>>(
                sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_z,  bb.idist, bb.fac, ba.y, ba.disp_y, ba.z, ba.disp_z);

        kernel_cal_surf_coff_F<<<grid_size, block_size>>>
        (ba.NN, bb.NL, bb.scr, ba.x, fncst_x, ba.y, fncst_y, ba.z, fncst_z,bb.idist);
        CHECK(cudaFree(fncst_x));
        CHECK(cudaFree(fncst_y));
        CHECK(cudaFree(fncst_z));
       ;
    }
}

void vol_Corr(int *NN, int *NL, real *x, real *idist, real *fac, real *y, real *z)
{

    for (int i=0; i<p.N; i++)
    {
        for (int j=0; j<NN[i]; j++)
        {
            int idx = i*p.MN +j;
            int cnode = NL[idx];
            if (y==NULL && z==NULL)
                idist[idx] = sqrt(square(x[cnode]-x[i]));
            if (y!=NULL && z==NULL)
                idist[idx] = sqrt(square(x[cnode]-x[i]) + square(y[cnode]-y[i]));
            if (y!=NULL && z!=NULL)
                idist[idx] = sqrt(square(x[cnode]-x[i]) + square(y[cnode]-y[i]) + square(z[cnode]-z[i]));

            if (idist[idx] <= p.delta-p.dx/2.0)
                fac[idx] = 1.0;
            else if (idist[idx] <= p.delta+p.dx/2.0)
                fac[idx] = (p.delta+p.dx/2.0-idist[idx]) / p.dx;
            else
                fac[idx] = 0.0;

        }
    }
}

static void Disp
        (real *coord, real *disp_cal, real *disp_initial_1=NULL, real *disp_initial_2=NULL)
{

    for (int i=0; i<p.N; i++)
    {
        disp_cal[i] = 0.001 * coord[i];
        if(disp_initial_1!=NULL)
            disp_initial_1[i] = 0.0;
        if(disp_initial_2!=NULL)
            disp_initial_2[i] = 0.0;
    }
}

static  void surface_F
        (real sedload_Cal, int *NN, int *NL, real *x, real *disp_x, real *fncst_Cal,
         real *idist, real *fac, real *y=NULL, real *disp_y=NULL, real *z=NULL, real *disp_z=NULL )
{
    int  cnode = 0;
    real nlength =0.0;

    int idx = 0;
    for (int i=0; i<p.N; i++)
    {
        real stendens_Cal = 0.0;
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*p.MN + j;
            cnode = NL[idx];
            if (y==NULL && z==NULL)
            {
                nlength = sqrt(square(x[cnode]+disp_x[cnode]-x[i]-disp_x[i]));
            }
            if (y!=NULL && z==NULL)
            {
                nlength = sqrt(square(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               square(y[cnode]+disp_y[cnode]-y[i]-disp_y[i]));
            }
            if (y!=NULL && z!=NULL)
            {
                nlength = sqrt(square(x[cnode]+disp_x[cnode]-x[i]-disp_x[i])+
                               square(y[cnode]+disp_y[cnode]-y[i]-disp_y[i])+
                               square(z[cnode]+disp_z[cnode]-z[i]-disp_z[i]));
            }
            stendens_Cal += 0.25*p.bc*(nlength-idist[idx])*(nlength-idist[idx])
                            /idist[idx]*p.vol*fac[idx];
        }
        fncst_Cal[i] = sedload_Cal / stendens_Cal;
    }
}

static void cal_surf_coff_F
        (int *NN, int *NL, real *scr, real *x, real *fncst_x,
         real *y=NULL, real *fncst_y=NULL,  real *z=NULL, real *fncst_z=NULL, real *idist=NULL)
{
    int  cnode = 0;
    real nlength =0.0;
    real stendens_Cal = 0.0;
    int idx = 0;

    for (int i=0; i<p.N; i++)
    {
        for (int j=0; j<NN[i]; j++)
        {
            idx = i*p.MN + j;
            cnode = NL[idx];

            if(y == NULL)
            {
                scr[idx] = (fncst_x[i] + fncst_x[cnode]) / 2.0;

            }
            if (y != NULL && z==NULL)
            {
                real theta = 0.0;
                real scx = 0.0;
                real scy = 0.0;
                if (fabs(y[cnode] - y[i]) <= 1.0e-10)
                    theta = 0.0;
                else if (fabs(x[cnode] - x[i]) <= 1.0e-10)
                    theta = 90.0*p.pi/180.0;
                else
                    theta = atan(fabs(y[cnode] - y[i])/fabs(x[cnode] - x[i]));
                scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
                scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;

                scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy)));
            }
            if (y != NULL && z!=NULL)
            {
                real theta = 0.0;
                real scx = 0.0;
                real scy = 0.0;
                real scz = 0.0;
                if(fabs(z[cnode]-z[i])<=1.0e-10)
                {
                    if(fabs(y[cnode]-y[i])<=1.0e-10)
                        theta = 0.0;
                    else if (fabs(x[cnode]-x[i])<=1.0e-10)
                        theta = 90.0*p.pi/180.0;
                    else
                        theta = atan(fabs(y[cnode] - y[i])/fabs(x[cnode] - x[i]));
                    real phi = 90.0*p.pi/180.0;
                    scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
                    scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
                    scz = (fncst_z[i] + fncst_z[cnode]) / 2.0;
                    scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy) +
                                         cos(phi)*cos(phi)/(scz*scz)));
                }
                else if (fabs(x[cnode]-x[i])<=1.0e-10 && fabs(y[cnode]-y[i])<=1.0e-10)
                    scr[idx] = (fncst_z[i] + fncst_z[cnode]) / 2.0;
                else
                {
                    theta = atan(fabs(y[cnode] - y[i])/fabs(x[cnode] - x[i]));
                    real phi = acos(fabs(z[cnode]-z[i])/idist[idx]);
                    scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
                    scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
                    scz = (fncst_z[i] + fncst_z[cnode]) / 2.0;
                    scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy) +
                                         cos(phi)*cos(phi)/(scz*scz)));
                }
            }
        }
    }
}

void surface_correct_cpu(Dselect &d, BAtom &ba, BBond &bb)
{
    real sedload_Cal = 0.0;
    if(d.Dim==1)
    {
        vol_Corr(ba.NN, bb.NL, ba.x, bb.idist, bb.fac);
        real *fncst_x = (real *) malloc(p.N*sizeof(real));
        Disp(ba.x, ba.disp_x);
        surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_x,  bb.idist, bb.fac);
        cal_surf_coff_F(ba.NN, bb.NL, bb.scr, ba.x, fncst_x);
        free(fncst_x);
    } else if(d.Dim==2)
    {
        sedload_Cal = 0.5*p.emod/(1.0-p.pratio*p.pratio) * 1.0e-6;
        vol_Corr(ba.NN, bb.NL, ba.x, bb.idist, bb.fac, ba.y);
        real *fncst_x = (real *) malloc(p.N*sizeof(real));
        real *fncst_y = (real *) malloc(p.N*sizeof(real));


        Disp(ba.x, ba.disp_x, ba.disp_y);

        surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_x,  bb.idist, bb.fac, ba.y, ba.disp_y);

        Disp(ba.y, ba.disp_y, ba.disp_x);

        surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_y,  bb.idist, bb.fac, ba.y, ba.disp_y);

        cal_surf_coff_F(ba.NN, bb.NL, bb.scr, ba.x, fncst_x, ba.y, fncst_y);


        free(fncst_x);
        free(fncst_y);


    } else
    {
        sedload_Cal =0.6*p.emod*1.0e-6;
        vol_Corr(ba.NN, bb.NL, ba.x, bb.idist, bb.fac, ba.y, ba.z);
        real *fncst_x = (real *) malloc(p.N*sizeof(real));
        real *fncst_y = (real *) malloc(p.N*sizeof(real));
        real *fncst_z = (real *) malloc(p.N*sizeof(real));

        Disp(ba.x, ba.disp_x, ba.disp_y, ba.disp_z);

        surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_x,  bb.idist, bb.fac, ba.y, ba.disp_y, ba.z, ba.disp_z);

        Disp(ba.y, ba.disp_y, ba.disp_x, ba.disp_z);

        surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_y,  bb.idist, bb.fac, ba.y, ba.disp_y, ba.z, ba.disp_z);
        Disp(ba.z, ba.disp_z, ba.disp_x, ba.disp_y);
        surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, fncst_z,  bb.idist, bb.fac, ba.y, ba.disp_y, ba.z, ba.disp_z);

        cal_surf_coff_F(ba.NN, bb.NL, bb.scr, ba.x, fncst_x, ba.y, fncst_y,  ba.z, fncst_z, bb.idist);


        free(fncst_x);
        free(fncst_y);
        free(fncst_z);
    }
}