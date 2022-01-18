//
// Created by wxm on 2021/12/2.
//

#include "Coord.cuh"
__global__ void kernel_Initial_coord_plate(real *x, real *y)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    //确定质点积分区域质点坐标
    if (idx < dp.N)
    {
        x[idx] = (-dp.nx/2.0+0.5+idx%dp.nx) * dp.dx;
        y[idx] = (-dp.ny/2.0+0.5+idx/dp.nx) * dp.dx;
    }
}



void Initial_coord_plate_cpu(real *x, real *y)
{

    //确定质点积分区域质点坐标
    for (int i=0;i<p.ny;i++)
    {
        for( int j=0;j<p.nx;j++)
        {
            x[i*p.nx+j] = (-p.nx/2.0+0.5 + j) * p.dx;
            y[i*p.nx+j] = (-p.ny/2.0+0.5+ i) * p.dx;
        }
    }
}


__global__ void kernel_coord_plate_crack(real *x, real *y)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < dp.N_int+dp.nx*3)
    {
        x[idx] = (-dp.nx/2.0+0.5+idx%dp.nx) * dp.dx;
        y[idx] = (-dp.ny/2.0+0.5+idx/dp.nx) * dp.dx;
    } else if ( idx <dp.N){
        x[idx] = (-dp.nx/2.0+0.5+idx%dp.nx) * dp.dx;
        y[idx] = (-dp.ny/2.0-0.5-(idx-dp.N_int-dp.nx*3)/dp.nx) * dp.dx;
    }
}

void coord_plate_crack_cpu(real *x, real *y)
{
    for (int i=0;i<p.ny+3;i++)
    {
        for( int j=0;j<p.nx;j++)
        {

            x[i*p.nx+j] = (-p.nx/2.0+0.5 + j) * p.dx;
            y[i*p.nx+j] = (-p.ny/2.0+0.5+ i) * p.dx;
        }
    }
    for (int i=p.ny+3;i<p.ny+6;i++)
    {
        for( int j=0;j<p.nx;j++)
        {
            x[i*p.nx+j] = (-p.nx/2.0+0.5 + j) * p.dx;
            y[i*p.nx+j] = (p.ny/2.0 + 3-0.5-i) * p.dx;
        }
    }
}




void Gen_Coord_plate_gpu( BAtom &ba)
{
    int grid_size = (p.N-1)/block_size + 1;
    kernel_Initial_coord_plate<<<grid_size,block_size>>>(ba.x,ba.y);
}

int coord_KW(real *x, real *y, real *z)
{
    real tx;
    real ty;
    real tz;
    unsigned int num=-1;
    for (int k=0; k<p.nz;k++)
    {
        for(int j=0; j<p.ny;j++)
        {
            for(int i=0; i<p.nx; i++)
            {
                tx = (-(p.nx-1)/2.0 + i) * p.dx;
                ty = (-(p.ny-1)/2.0+ j) * p.dx;
                tz = (-p.nz/2.0+0.5 + k) * p.dx;
                //if(i==1)  cout<<"x "<<tx<<" y "<<ty<<" z "<<tz<<endl;
                //if(i==0 && j==0)  cout<<"x "<<p.dx<<" y "<<p.nx<<endl;
                if ((tx>(-cyp.rad-1.1*p.dx) && tx<(-cyp.rad+0.1*p.dx)) )
                {
                    if(ty<-1.0e-10)
                    {
                        num++;

                        x[num] = tx;
                        y[num] = ty;
                        z[num] = tz;
                    }
                }
                else if ( tx>(cyp.rad-0.1*p.dx) && tx<(cyp.rad+1.1*p.dx))
                {
                    if(ty<-1.0e-10)
                    {
                        num++;

                        x[num] = tx;
                        y[num] = ty;
                        z[num] = tz;
                    }
                }
                else
                {
                    num++;

                    x[num] = tx;
                    y[num] = ty;
                    z[num] = tz;
                }
            }
        }
    }

    return num;
}



