//
// Created by wxm on 2021/12/3.
//

#include "Initial.cuh"
static __global__ void kernel_initial_fail(int *fail)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;

    if (idx<dp.N_int*dp.MN)
    {
        fail[idx] = 1;
    }
}

inline void initial_fail_gpu(BBond &bb)
{

    int grid_size = (p.N_int*p.MN-1)/block_size + 1;
    kernel_initial_fail<<<grid_size,block_size>>>(bb.fail);

}

void base_integrate_initial_gpu(Dselect &d,BAtom &ba, BBond &bb)
{

    size_t byte = p.N*sizeof(real);

    initial_fail_gpu(bb);
    CHECK(cudaMemset(ba.disp_x, 0, byte));
    CHECK(cudaMemset(ba.acc_x, 0, byte));
    CHECK(cudaMemset(ba.vel_x, 0, byte));
    CHECK(cudaMemset(ba.bforce_x, 0, byte));
    CHECK(cudaMemset(ba.pforce_x, 0, byte));

    if (d.Dim==2 | d.Dim==3)
    {
        CHECK(cudaMemset(ba.disp_y, 0, byte));
        CHECK(cudaMemset(ba.acc_y, 0, byte));
        CHECK(cudaMemset(ba.vel_y, 0, byte));
        CHECK(cudaMemset(ba.bforce_y, 0, byte));
        CHECK(cudaMemset(ba.pforce_y, 0, byte));


        if(d.Dim==3)
        {
            CHECK(cudaMemset(ba.disp_z, 0, byte));
            CHECK(cudaMemset(ba.acc_z, 0, byte));
            CHECK(cudaMemset(ba.vel_z, 0, byte));
            CHECK(cudaMemset(ba.bforce_z, 0, byte));
            CHECK(cudaMemset(ba.pforce_z, 0, byte));

        }
    }
}

void static_initial_bond_gpu(Dselect &d,BAtom &ba, BBond &bb)
{

    size_t byte = p.N*sizeof(real);
    base_integrate_initial_gpu(d,ba, bb);

    CHECK(cudaMemset(ba.velhalfold_x, 0, byte));
    CHECK(cudaMemset(ba.pforceold_x, 0, byte));
    if (d.Dim==2 | d.Dim==3)
    {
        CHECK(cudaMemset(ba.velhalfold_y, 0, byte));
        CHECK(cudaMemset(ba.pforceold_y, 0, byte));
        if(d.Dim==3)
        {
            CHECK(cudaMemset(ba.velhalfold_z, 0, byte));
            CHECK(cudaMemset(ba.pforceold_z, 0, byte));
        }
    }
}

static void initial_fail(int *fail)
{
    for (int i=0; i<p.N_int*p.MN; i++) fail[i] = 1;
}

void base_integrate_initial_cpu(Dselect &d,BAtom &ba, BBond &bb)
{

    size_t byte = p.N*sizeof(real);

    initial_fail(bb.fail);
    memset(ba.disp_x, 0, byte);
    memset(ba.vel_x, 0, byte);
    memset(ba.acc_x, 0, byte);
    memset(ba.bforce_x, 0, byte);



    if (d.Dim==2 | d.Dim==3)
    {
        memset(ba.disp_y, 0, byte);
        memset(ba.acc_y, 0, byte);
        memset(ba.vel_y, 0, byte);
        memset(ba.bforce_y, 0, byte);


        if(d.Dim==3)
        {
            memset(ba.disp_z, 0, byte);
            memset(ba.acc_z, 0, byte);
            memset(ba.vel_z, 0, byte);
            memset(ba.bforce_z, 0, byte);

        }
    }
}

void static_initial_bond_cpu(Dselect &d,BAtom &ba, BBond &bb)
{

    size_t byte = p.N*sizeof(real);
    base_integrate_initial_cpu(d,ba, bb);

    memset(ba.velhalfold_x, 0, byte);
    memset(ba.pforceold_x, 0, byte);
    if (d.Dim==2 | d.Dim==3)
    {
        memset(ba.velhalfold_y, 0, byte);
        memset(ba.pforceold_y, 0, byte);
        if(d.Dim==3)
        {
            memset(ba.velhalfold_z, 0, byte);
            memset(ba.pforceold_z, 0, byte);
        }
    }
}


