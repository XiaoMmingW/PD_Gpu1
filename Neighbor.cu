//
// Created by wxm on 2021/12/2.
//

#include "Neighbor.cuh"

__global__ void  kernel_find_neighbor_2D
(const real *x, const real *y, int *NN, int *NL)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<dp.N)
    {

        for (int j=i+1; j<dp.N; ++j)
        {
            if ( gsquare(x[j]-x[i])+gsquare(y[j]-y[i]) < dp.delta*dp.delta)
            {
                NL[i * dp.MN + atomicAdd(&NN[i],1)] = j;
                NL[j * dp.MN + atomicAdd(&NN[j],1)] = i;
            }
        }
    }
}

__global__ void  kernel_find_neighbor_3D
(const real *x, const real *y, const real *z,int *NN, int *NL)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<dp.N)
    {
        for (int j=i+1; j<dp.N; ++j)
        {
            if ( gsquare(x[j]-x[i])+gsquare(y[j]-y[i]) + gsquare(z[j]-z[i]) < dp.delta*dp.delta)
            {
                NL[i * dp.MN + atomicAdd(&NN[i],1)] = j;
                NL[j * dp.MN + atomicAdd(&NN[j],1)] = i;
            }
        }
    }
}

void find_neighbor_2D(const real *x, const real *y, int *NN, int *NL)
{
    for (int i=0;i<p.N;i++)
        NN[i] = 0;

    for (int i=0;i<p.N;i++)
    {
        for (int j=i+1; j<p.N; ++j)
        {
            if (square(x[j]-x[i])+square(y[j]-y[i]) < p.delta*p.delta)
            {
                NL[i * p.MN + NN[i]++] = j;
                NL[j * p.MN + NN[j]++] = i;
              //  if (NN[i]==p.MN | NN[j]==p.MN) break;
            }
        }
    }
}



void find_neighbor_3D(const real *x, const real *y, const real *z, int *NN, int *NL)
{
    for (int i=0;i<p.N;i++)
        NN[i] = 0;

    for (int i=0;i<p.N;i++)
    {
        for (int j=i+1; j<p.N; ++j)
        {
            if (square(x[j]-x[i])+square(y[j]-y[i]) +square(z[j]-z[i]) < p.delta*p.delta)
            {
                NL[i * p.MN + NN[i]++] = j;
                NL[j * p.MN + NN[j]++] = i;
            }
           //if (NN[i]==p.MN | NN[j]==p.MN) break;
        }
    }
}

void find_neighbor_gpu(Dselect &d, BAtom &ba, BBond &bb) {

    int grid_size = (p.N-1)/block_size + 1;
    if(d.Dim==2){
        kernel_find_neighbor_2D<<<grid_size,block_size>>>(ba.x, ba.y,ba.NN,bb.NL);
    } else {
        kernel_find_neighbor_3D<<<grid_size,block_size>>>(ba.x, ba.y, ba.z,ba.NN,bb.NL);
    }
}