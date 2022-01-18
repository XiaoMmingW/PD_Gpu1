//
// Created by wxm on 2021/12/5.
//

#include "Solve_Plate_Tensile.cuh"

static __global__ void load_plate_tensile_gpu(real *bforce_x)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<dp.ny)
    {
        bforce_x[i*dp.nx] = -0.001*dp.emod/dp.dx;
    } else if (i<dp.ny*2 && i>=dp.ny)
    {
        bforce_x[(i-dp.ny)*dp.nx+dp.nx-1] = 0.001*dp.emod/dp.dx;
    }
}

static void load_plate_tensile_cpu(real *bforce_x)
{
    for (int i=0; i<p.ny; i++)
    {
        bforce_x[i*p.nx] = -0.001*p.emod/p.dx;
        bforce_x[i*p.nx+p.nx-1] = 0.001*p.emod/p.dx;
    }
}



void solve_plate_tensile(Dselect &d, real dx)
{
    BAtom ba;
    BBond bb;
    d.Dim = 2;
    d.B_S = 1;
    d.Mode =1;
    real f_dx = 0.01;
    if (dx>1.0e-8)
        f_dx = dx;
    BaseModel_Parameter bp(d, 1.0, 1000, f_dx,  1.0,0.01,
                           1.0, 0.5, 0.0,200.e9,1./3., 7850,1.0);

    initialil_parameter(p, bp);
    cout<<"nx "<<p.nx<<" ny "<<p.ny<<endl;

    Base_Allocate(ba, bb, d);

    if(d.device==1)
    {
        long double start = cpuSecond();
        long double start1 = cpuSecond();
        Initial_coord_plate_cpu(ba.x, ba.y);
        cout<<"time coord "<<cpuSecond() - start<<endl;
        cout<<"neighbor"<<endl;
        start = cpuSecond();
        find_neighbor_2D(ba.x, ba.y, ba.NN, bb.NL);
        cout<<"time neighbor "<<cpuSecond() - start<<endl;
        cout<<"neighbor"<<endl;
        start = cpuSecond();
        surface_correct_cpu(d, ba, bb);
        cout<<"time surface "<<cpuSecond() - start<<endl;
        start = cpuSecond();
        static_initial_bond_cpu(d,ba, bb);
        cout<<"time initial "<<cpuSecond() - start<<endl;
        start = cpuSecond();
        load_plate_tensile_cpu(ba.bforce_x);
        cout<<"time load "<<cpuSecond() - start<<endl;
        long double force_time = 0.0;
        long double integrate_time = 0.0;

        for (int i=0; i<p.nt; i++)
        {
            start = cpuSecond();
            bond_force_cpu(
                    ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y, ba.NN, bb.NL,
                    bb.idist, bb.scr, bb.fail, bb.fac);
            force_time += cpuSecond() - start;
            start = cpuSecond();
            static_integrate_cpu(
                    i, ba.pforce_x, ba.velhalf_x, ba.bforce_x, ba.vel_x,  ba.velhalfold_x,  ba.disp_x,
                    ba.pforceold_x, ba.pforce_y, ba.velhalf_y, ba.bforce_y,  ba.vel_y, ba.velhalfold_y,
                    ba.disp_y,  ba.pforceold_y);
            integrate_time += cpuSecond() - start;
        //cout<<i<<endl;
        }
        cout<<"time force "<<force_time<<endl;
        cout<<"time integrate "<<integrate_time<<endl;
        cout<<"time "<<cpuSecond() - start1<<endl;
        save_disp_cpu(d, ba, "disp.txt");
    } else
    {

        cudaMemcpyToSymbol(dp, &p,sizeof(device_parameter));
        int grid_size1 = (p.N-1)/block_size + 1;
        unsigned int grid_size2 = (p.N*p.MN-1)/block_size + 1;
        int grid_size3 = (p.ny*2 - 1)/block_size + 1;
        long double start = cpuSecond();
        kernel_Initial_coord_plate<<<grid_size1,block_size>>>(ba.x,ba.y);
        kernel_find_neighbor_2D<<<grid_size1,block_size>>>(ba.x, ba.y,ba.NN,bb.NL);

        surface_correct_gpu(d, ba, bb);

        static_initial_bond_gpu(d,ba, bb);
        load_plate_tensile_gpu<<<grid_size3,block_size>>>(ba.bforce_x);

        for (int i=0; i<p.nt; i++)
        {
            cout<<i<<endl;
            kernel_particle_force_2D<<<grid_size1,block_size>>>(
                    ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y, ba.NN, bb.NL,
                    bb.idist, bb.scr, bb.fail, bb.fac);
            //CHECK(cudaDeviceSynchronize());
           // cout<<"2"<<endl;
//            kernel_bond_force_2D<<<grid_size2,block_size>>>(
//                    ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y, ba.NN, bb.NL,
//                    bb.idist, bb.scr, bb.fail, bb.fac);

            kernel_cnn<<<grid_size1, block_size>>>(
                    ba.cn_xy,ba.pforce_x, ba.pforceold_x,  ba.velhalfold_x,
                    ba.disp_x, ba.disp_xy, ba.pforce_y, ba.pforceold_y,
                    ba.velhalfold_y, ba.disp_y);

            real cn = cal_cn(ba);

            kernel_integrate<<<grid_size1, block_size>>>(
                    i, cn,  ba.pforce_x, ba.velhalf_x, ba.bforce_x,
                    ba.vel_x,  ba.velhalfold_x,  ba.disp_x, ba.pforceold_x, ba.pforce_y,
                    ba.velhalf_y, ba.bforce_y,  ba.vel_y, ba.velhalfold_y,
                    ba.disp_y,  ba.pforceold_y);

        }
        CHECK(cudaDeviceSynchronize());
        cout<<"time "<<cpuSecond() - start<<endl;
        save_disp_gpu(d, ba, "disp.txt");
    }
    Base_Free(ba, bb, d);

}