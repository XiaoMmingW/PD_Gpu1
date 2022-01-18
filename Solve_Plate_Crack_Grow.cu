//
// Created by wxm on 2021/12/6.
//

#include "Solve_Plate_Crack_Grow.cuh"


static __global__ void kernel_quick_load(real load, real *disp_y)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<3*dp.nx)
    {
        disp_y[dp.N_int+i] += load*dp.dt;
    }
    else if(i<6*dp.nx)
    {
        disp_y[dp.N_int+i] -= load*dp.dt;
    }
}
void quick_load_cpu(real load, real *disp_y)
{
    for (int i=0; i<3*p.nx; i++)
    {
        disp_y[p.N_int+i] += load*p.dt;
        disp_y[p.N_int+3*p.nx+i] -= load*p.dt;
    }
}


void solve_plate_crack_grow(Dselect &d,real dx)
{
    BAtom ba;
    BBond bb;
    d.Dim = 2;
    d.B_S = 2;
    d.Mode = 2;
    real f_dx = 0.0001;
    if (dx>1.0e-8)
        f_dx = dx;
    BaseModel_Parameter bp(d, 1.3367e-8, 1250, f_dx,  1.0,0.0001,
                           0.05, 0.05, 0.0,192.e9,0.3, 7850,0.04472);
    bp.N += 6*bp.nx;
    initialil_parameter(p, bp);
    cout<<"nx "<<p.nx<<" ny "<<p.ny<<endl;

    Base_Allocate(ba, bb, d);

    if(d.device==1)
    {
        long double start = cpuSecond();
        long double start1 = cpuSecond();
        start = cpuSecond();
        coord_plate_crack_cpu(ba.x, ba.y);
        cout<<"time COORD "<<cpuSecond() - start<<endl;
        cout<<"neighbor"<<endl;
        start = cpuSecond();
        find_neighbor_2D(ba.x, ba.y, ba.NN, bb.NL);
        cout<<"neighbor"<<endl;
        cout<<"time neighbor "<<cpuSecond() - start<<endl;
        start = cpuSecond();
        base_integrate_initial_cpu(d, ba, bb);
        cout<<"time initial "<<cpuSecond() - start<<endl;
        start = cpuSecond();
        set_crack_2D_cpu(0.01, 0.0, 0.0, 0.0, ba.NN, bb.fail, bb.NL, ba.x, ba.y);
        cout<<"time crack "<<cpuSecond() - start<<endl;
        start = cpuSecond();
        vol_Corr(ba.NN, bb.NL, ba.x, bb.idist, bb.fac,ba.y);
        initial_weight_cpu(ba.m, ba.NN, bb.w, bb.idist, bb.fac);
        cout<<"time weight "<<cpuSecond() - start<<endl;
        long double force_time = 0.0;
        long double integrate_time = 0.0;
        long double load_time = 0.0;

        for (int i=0; i<p.nt; i++)
        {
            start = cpuSecond();
            quick_load_cpu(20, ba.disp_y);
            load_time += cpuSecond() - start;
            start = cpuSecond();
            integrate_1(ba.vel_x, ba.acc_x, ba.disp_x, ba.vel_y, ba.acc_y, ba.disp_y);
            integrate_time += cpuSecond() - start;
            start = cpuSecond();
            state_force_2D_cpu( ba.NN, ba.m, ba.theta,  bb.fail, bb.NL, bb.e, bb.w, bb.idist,
                            bb.fac,ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y,ba.pforce_y);
            force_time += cpuSecond() - start;
            start = cpuSecond();
            integrate_2(ba.vel_x, ba.acc_x, ba.pforce_x, ba.bforce_x, ba.vel_y, ba.acc_y, ba.pforce_y, ba.bforce_y);
            integrate_time += cpuSecond() - start;
        }
        start = cpuSecond();
        cal_dmg_cpu(ba.NN,ba.dmg, bb.fail, bb.fac);
        cout<<"time dmg"<<cpuSecond() - start<<endl;
        cout<<"time force "<<force_time<<endl;
        cout<<"time integrate "<<integrate_time<<endl;
        cout<<"time load "<<load_time<<endl;
        cout<<"time "<<cpuSecond() - start1<<endl;
        save_disp_cpu(d, ba, "disp.txt");
    } else
    {
        cudaMemcpyToSymbol(dp, &p,sizeof(device_parameter));
        unsigned int grid_size1 = (p.N-1)/block_size + 1;
        unsigned int grid_size2 = (p.N*p.MN-1)/block_size + 1;
        unsigned int grid_size3 = (p.N_int - 1)/block_size + 1;
        unsigned int grid_size4 = (p.N_int*p.MN-1)/block_size + 1;
        long double start = cpuSecond();
        kernel_coord_plate_crack<<<grid_size1,block_size>>>(ba.x, ba.y);
        CHECK(cudaMemset(ba.NN,0,p.N* sizeof(int)));
        kernel_find_neighbor_2D<<<grid_size1,block_size>>>(ba.x, ba.y,ba.NN,bb.NL);

        base_integrate_initial_gpu(d,ba, bb);

        kernel_set_crack_2D<<<grid_size4,block_size>>>(
               0.01, 0.0, 0.0, 0.0, ba.NN, bb.fail, bb.NL, ba.x, ba.y);

        kernel_vol_Corr<<<grid_size2, block_size>>>(ba.NN, bb.NL, ba.x, bb.idist, bb.fac, ba.y);

        kernel_initial_weight_2D<<<grid_size2, block_size>>>(
                ba.m, ba.NN, bb.w, bb.idist, bb.fac);
        //real *disp_x = (real*) malloc(p.N* sizeof(real));
        //CHECK(cudaMemcpy(disp_x, ba.m, p.N* sizeof(real), cudaMemcpyDeviceToHost));
        //cout<<"m "<<disp_x[p.N-1]<<endl;
        cout<<"N "<<p.N<<endl;
 //       real *dmg = new real[p.nx];
//        ofstream ofs;
//        ofs.open("grow.txt", ios::out);
        for (int i=0; i<p.nt; i++)
        {
            kernel_quick_load<<<(6*p.nx-1)/block_size+1,block_size>>>(20, ba.disp_y);
            kernel_integrate_2D_1<<<grid_size3,block_size>>>(
                    ba.vel_x, ba.acc_x, ba.disp_x, ba.vel_y, ba.acc_y, ba.disp_y);
            //CHECK(cudaMemcpy(disp_x, ba.theta, p.N* sizeof(real), cudaMemcpyDeviceToHost));
            //cout<<"m "<<disp_x[0]<<endl;
            kernel_cal_theta_2D<<<grid_size2, block_size>>>(
                    ba.NN, ba.m, ba.theta, bb.NL, bb.w, bb.fail, bb.e, bb.idist,bb.fac,
                    ba.x, ba.disp_x, ba.y, ba.disp_y);

            kernel_state_force_2D<<<grid_size4, block_size>>>(
                    ba.NN, ba.m, ba.theta,  bb.fail, bb.NL, bb.e, bb.w, bb.idist,
                    bb.fac,ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y,ba.pforce_y);
//            kernel_cal_theta_2D_particle<<<grid_size1, block_size>>>(
//                    ba.NN, ba.m, ba.theta, bb.NL, bb.w, bb.fail, bb.e, bb.idist,bb.fac,
//                    ba.x, ba.disp_x, ba.y, ba.disp_y);
//
//            kernel_state_force_2D_particle<<<grid_size3, block_size>>>(
//                    ba.NN, ba.m, ba.theta,  bb.fail, bb.NL, bb.e, bb.w, bb.idist,
//                    bb.fac,ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y,ba.pforce_y);

            kernel_integrate_2D_2<<<grid_size3,block_size>>>
            (ba.vel_x, ba.acc_x, ba.pforce_x, ba.bforce_x, ba.vel_y, ba.acc_y, ba.pforce_y, ba.bforce_y);
//            if (i%20==0)
//            {
//                kernel_cal_dmg_2D<<<grid_size4,block_size>>>(ba.NN,ba.dmg, bb.fail, bb.fac);
//                CHECK(cudaMemcpy(dmg, &ba.dmg[p.ny/2*p.nx],p.nx* sizeof(real), cudaMemcpyDeviceToHost));
//                for (int k=0;k<p.nx/2;k++)
//                {
//                    if(dmg[k]>0.38)
//                    {
//                        ofs<<(p.nx/2-k-50)*p.dx<<" "<<i*p.dt<<endl;
//                        break;
//                    }
//                }
           // }
        }
        kernel_cal_dmg_2D<<<grid_size4,block_size>>>(ba.NN,ba.dmg, bb.fail, bb.fac);
        CHECK(cudaDeviceSynchronize());
        cout<<"time "<<cpuSecond() - start<<endl;
        save_disp_gpu(d, ba, "disp.txt");
    }
    Base_Free(ba, bb, d);

}