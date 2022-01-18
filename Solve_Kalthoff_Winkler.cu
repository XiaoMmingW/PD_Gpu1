//
// Created by wxm on 2021/12/7.
//

#include "Solve_Kalthoff_Winkler.cuh"



__global__ void kernel_initial_cylinder(real *vel, real *disp, real *acc, real *force)
{
    unsigned int i = threadIdx.x;
    if(i<3)
    {
        acc[i] = 0.0;
        disp[i] =0.0;
        vel[i] = 0.0;
        force[i] = 0.0;
        if(i==1)
            vel[i] = dcyp.v;
        //printf("vel %f\n",vel[1]);
    }
}

void initial_cylinder_cpu(real *vel, real *disp, real *acc, real *force)
{

    for(int i=0;i<3;i++)
    {
        acc[i] = 0.0;
        disp[i] =0.0;
        vel[i] = 0.0;
        force[i] = 0.0;
        if(i==1)
            vel[i] = cyp.v;
        //printf("vel %f\n",vel[1]);
    }
}


__global__ void kernel_set_crack(int *NN, int *NL, int *fail, real *x, real*y)
{
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int i = idx/dp.MN;
    unsigned int j = idx%dp.MN;
    unsigned int cnode = 0;
    if (i<dp.N_int)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            if((x[cnode]>-dcyp.rad && x[i]< -dcyp.rad) | (x[i]>-dcyp.rad && x[cnode]< -dcyp.rad))
            {
                if (y[cnode]>=-1.0e-10 | y[i]>=-1.0e-10)
                    fail[idx] = 0;
            }
            if( (x[cnode]>dcyp.rad && x[i]< dcyp.rad) | (x[i]>dcyp.rad && x[cnode]< dcyp.rad))
            {
                if (y[cnode]>=-1.0e-10 | y[i]>=-1.0e-10)
                    fail[idx] = 0;
            }
        }
    }
}

void set_crack_cpu(int *NN, int *NL, int *fail, real *x, real*y)
{
    unsigned int cnode = 0;
    unsigned int idx = 0;
    for (int i=0; i<p.N_int; i++)
    {
        for (int j=0; j<NN[i];j++)
        {
            idx = i*p.MN + j;
            cnode = NL[idx];
            if((x[cnode]>-cyp.rad && x[i]< -cyp.rad) | (x[i]>-cyp.rad && x[cnode]< -cyp.rad))
            {
                if (y[cnode]>=-1.0e-10 | y[i]>=-1.0e-10)
                    fail[idx] = 0;
            }
            if( (x[cnode]>cyp.rad && x[i]< cyp.rad) | (x[i]>cyp.rad && x[cnode]< cyp.rad))
            {
                if (y[cnode]>=-1.0e-10 | y[i]>=-1.0e-10)
                    fail[idx] = 0;
            }
        }
    }
}

__global__ void kernel_cy_integrate_1(
        real *vel, real *disp, real *acc)
{
    unsigned int i = threadIdx.x;
    if(i<3)
    {
        vel[i] += (dp.dt/2.0) * acc[i];
        disp[i] += vel[i]*dp.dt;
       // if (i==1) printf("vel %f %f\n",vel[i], disp[1]);
    }
}

void cy_integrate_1(
        real *vel, real *disp, real *acc)
{
    for(int i=0;i<3;i++)
        {
            vel[i] += (p.dt/2.0) * acc[i];
            disp[i] += vel[i]*p.dt;
          //  if (i==1) printf("vel %f %f\n",vel[i], disp[1]);
        }
}

__global__ void kernel_contact(
        int loc_begin, real *dmg,real *x, real *disp_x, real *vel_x, real *y, real *disp_y, real *vel_y,
        real *z, real *disp_z, real *vel_z, real *force, real *disp)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<dp.N)
    {

        //i += loc_begin;
        if(dmg[i]<1.0)
        {
            real dprx = 0.0;
            real dpry = 0.0;
            real dprz = 0.0;
            real olddisp_x = 0.0;
            real olddisp_y = 0.0;
            real olddisp_z = 0.0;
            real oldvel_x = 0.0;
            real oldvel_y = 0.0;
            real oldvel_z = 0.0;
            real dpenx = x[i] + disp_x[i] - dcyp.x - disp[0];
            real dpeny = y[i] + disp_y[i] - dcyp.y - disp[1];
            real dpenz = z[i] + disp_z[i] - dcyp.z - disp[2];
            real dpend = sqrt(dpenx * dpenx + dpenz * dpenz);
            if (dpend <= dcyp.rad && fabs(dpeny)<=(dcyp.height/2.0))
            {
                //printf("contact\n");
                if(dcyp.rad-dpend <= dcyp.height/2.0 - fabs(dpeny))
                {
                    dprx = dpenx / dpend * dcyp.rad;
                    dpry = dpeny;
                    dprz = dpenz / dpend * dcyp.rad;
                } else if (dpeny >= 0.0){
                    dprx = dpenx;
                    dpry = dcyp.height/2.0;
                    dprz = dpenz;
                } else {
                    dprx = dpenx;
                    dpry = -dcyp.height/2.0;
                    dprz = dpenz;
                }

                olddisp_x = disp_x[i];
                olddisp_y = disp_y[i];
                olddisp_z = disp_z[i];
                oldvel_x = vel_x[i];
                oldvel_y = vel_y[i];
                oldvel_z = vel_z[i];
                disp_x[i] = dcyp.x + disp[0] + dprx - x[i];
                disp_y[i] = dcyp.y + disp[1] + dpry - y[i];
                disp_z[i] = dcyp.z + disp[2] + dprz - z[i];
                vel_x[i] = (disp_x[i]-olddisp_x)/dp.dt;
                vel_y[i] = (disp_y[i]-olddisp_y)/dp.dt;
                vel_z[i] = (disp_z[i]-olddisp_z)/dp.dt;

                atomicAdd(&force[0],-dp.dens*(vel_x[i]-oldvel_x)/dp.dt*dp.vol);
                atomicAdd(&force[1],-dp.dens*(vel_y[i]-oldvel_y)/dp.dt*dp.vol);
                atomicAdd(&force[2],-dp.dens*(vel_z[i]-oldvel_z)/dp.dt*dp.vol);

            }
        }
    }
}

void contact(
        int loc_begin, real *dmg,real *x, real *disp_x, real *vel_x, real *y, real *disp_y, real *vel_y,
        real *z, real *disp_z, real *vel_z, real *force, real *disp)
{


   for (int i=0; i<p.N_int; i++)
    {

        //i += loc_begin;
        if(dmg[i]<1.0)
        {
            real dprx = 0.0;
            real dpry = 0.0;
            real dprz = 0.0;
            real olddisp_x = 0.0;
            real olddisp_y = 0.0;
            real olddisp_z = 0.0;
            real oldvel_x = 0.0;
            real oldvel_y = 0.0;
            real oldvel_z = 0.0;
            real dpenx = x[i] + disp_x[i] - cyp.x - disp[0];
            real dpeny = y[i] + disp_y[i] - cyp.y - disp[1];
            real dpenz = z[i] + disp_z[i] - cyp.z - disp[2];
            real dpend = sqrt(dpenx * dpenx + dpenz * dpenz);
            if (dpend <= cyp.rad && fabs(dpeny)<=(cyp.height/2.0))
            {
                //printf("contact\n");
                if(cyp.rad-dpend <= cyp.height/2.0 - fabs(dpeny))
                {
                    dprx = dpenx / dpend * cyp.rad;
                    dpry = dpeny;
                    dprz = dpenz / dpend * cyp.rad;
                } else if (dpeny >= 0.0){
                    dprx = dpenx;
                    dpry = cyp.height/2.0;
                    dprz = dpenz;
                } else {
                    dprx = dpenx;
                    dpry = -cyp.height/2.0;
                    dprz = dpenz;
                }

                olddisp_x = disp_x[i];
                olddisp_y = disp_y[i];
                olddisp_z = disp_z[i];
                oldvel_x = vel_x[i];
                oldvel_y = vel_y[i];
                oldvel_z = vel_z[i];
                disp_x[i] = cyp.x + disp[0] + dprx - x[i];
                disp_y[i] = cyp.y + disp[1] + dpry - y[i];
                disp_z[i] = cyp.z + disp[2] + dprz - z[i];
                vel_x[i] = (disp_x[i]-olddisp_x)/p.dt;
                vel_y[i] = (disp_y[i]-olddisp_y)/p.dt;
                vel_z[i] = (disp_z[i]-olddisp_z)/p.dt;
                force[0] += -p.dens*(vel_x[i]-oldvel_x)/p.dt*p.vol;
                force[1] += -p.dens*(vel_y[i]-oldvel_y)/p.dt*p.vol;
                force[2] += -p.dens*(vel_z[i]-oldvel_z)/p.dt*p.vol;

            }
        }
    }
}


__global__ void kernel_cy_integrate_2(
        real *vel, real *acc, real *force)
{
    unsigned int i = threadIdx.x;
    if (i<3)
    {
        acc[i] = force[i]/dcyp.mass;
        vel[i] += dp.dt/2.0 * acc[i];
        force[i] = 0.0;

    }
}

void cy_integrate_2(
        real *vel, real *acc, real *force)
{
    for(int i=0; i<3; i++)
    {
        acc[i] = force[i]/cyp.mass;
        vel[i] += p.dt/2.0 * acc[i];
        force[i] = 0.0;

    }
}


void solve_kalthoff_winkler(Dselect &d, real dx)
{
    Cylinder cy;
    BAtom ba;
    BBond bb;
    d.Dim = 3;
    d.B_S = 1;
    d.Mode =2;
    real f_dx = 0.001;

    if (dx>1.0e-8)
        f_dx = dx;
    BaseModel_Parameter bp(d, 8.7e-8, 0, f_dx,  1.0,0.001,
                           0.2, 0.1, 0.009,191.0e9,0.25, 8000,0.01);
    bp.nx++;
    bp.ny++;
    bp.N = bp.nx*bp.ny*bp.nz;
    cyp.rad = 0.025;
    cyp.mass = 1.57;
    cyp.v = -32.0;
    cyp.height = 0.05;
    cyp.x = 0.0;
    cyp.y = bp.height/2.0 + cyp.height/2.0 + 0.1*bp.dx;
    cyp.z = 0.0;
    initialil_parameter(p, bp);
    real *x = (real *)malloc(p.N*sizeof(real));
    real *y = (real *)malloc(p.N*sizeof(real));
    real *z = (real *)malloc(p.N*sizeof(real));
    p.N = p.N_int = coord_KW(x, y, z)+1;
    cout<<"x "<<x[0]<<" y "<<y[0]<<endl;
    cout<<"N "<<p.N<<endl;
    cout<<"nx "<<p.nx<<" ny "<<p.ny<<" nz "<<p.nz<<endl;

    Base_Allocate(ba,bb, d);
    cy_allocate(d,cy);
    if (d.device==1)
    {
        CHECK(cudaMemcpy(ba.x, x, p.N * sizeof(real), cudaMemcpyHostToHost));
        CHECK(cudaMemcpy(ba.y, y, p.N * sizeof(real), cudaMemcpyHostToHost));
        CHECK(cudaMemcpy(ba.z, z, p.N * sizeof(real), cudaMemcpyHostToHost));
        long double start = cpuSecond();
        long double start1 = cpuSecond();
        cout<<"neighbor"<<endl;
        start = cpuSecond();
        find_neighbor_3D(ba.x, ba.y, ba.z, ba.NN, bb.NL);
        cout<<"time neighbor"<<cpuSecond() - start<<endl;
        cout<<"neighbor"<<endl;
        start = cpuSecond();
        surface_correct_cpu(d,ba,bb);
        cout<<"time surface"<<cpuSecond() - start<<endl;
        start = cpuSecond();
        base_integrate_initial_cpu(d,ba,bb);
        cout<<"time initial"<<cpuSecond() - start<<endl;
        start = cpuSecond();
        set_crack_cpu(ba.NN, bb.NL, bb.fail, ba.x, ba.y);
        cout<<"time crack"<<cpuSecond() - start<<endl;
        start = cpuSecond();
        initial_cylinder_cpu(cy.vel, cy.disp, cy.acc, cy.force);
        cout<<"time initial"<<cpuSecond() - start<<endl;
        long double force_time = 0.0;
        long double integrate_time = 0.0;
        long double contact_time = 0.0;
        long double integratecy_time = 0.0;

        for (int i=0; i<p.nt; i++)
        {
            cout<<i<<endl;
            start = cpuSecond();
            integrate_1(ba.vel_x, ba.acc_x, ba.disp_x, ba.vel_y, ba.acc_y, ba.disp_y,ba.vel_z, ba.acc_z, ba.disp_z );
            integrate_time += cpuSecond() - start;
            start = cpuSecond();
            cy_integrate_1(cy.vel, cy.disp, cy.acc);
            integratecy_time += cpuSecond() - start;
            start = cpuSecond();
            bond_force_cpu(
                    ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y, ba.NN, bb.NL,
                    bb.idist, bb.scr, bb.fail, bb.fac, ba.z, ba.disp_z, ba.pforce_z);
            force_time += cpuSecond() - start;
            start = cpuSecond();
            contact(0, ba.dmg,ba.x, ba.disp_x, ba.vel_x, ba.y, ba.disp_y, ba.vel_y,
                    ba.z, ba.disp_z, ba.vel_z, cy.force, cy.disp);
            contact_time += cpuSecond() - start;
            start = cpuSecond();
            integrate_2(ba.vel_x, ba.acc_x, ba.pforce_x, ba.bforce_x, ba.vel_y, ba.acc_y, ba.pforce_y, ba.bforce_y,
                        ba.vel_z, ba.acc_z, ba.pforce_z, ba.bforce_z );
            integrate_time += cpuSecond() - start;
            start = cpuSecond();
            cy_integrate_2(cy.vel, cy.acc, cy.force);
            integratecy_time += cpuSecond() - start;

        }
        start = cpuSecond();
        cal_dmg_cpu(ba.NN,ba.dmg, bb.fail, bb.fac);
        cout<<"time dmg"<<cpuSecond() - start<<endl;
        cout<<"time force "<<force_time<<endl;
        cout<<"time integrate "<<integrate_time<<endl;
        cout<<"time contact "<<contact_time<<endl;
        cout<<"time cy "<<integratecy_time<<endl;
        cout<<"time "<<cpuSecond() - start1<<endl;

        save_kw_cpu(ba, "disp.txt");
    }
    else
    {
        const int smem = block_size*sizeof(real);
        unsigned int loc_begin = p.N_int-p.nx*p.ny;
        cudaMemcpyToSymbol(dp, &p,sizeof(device_parameter));
        cudaMemcpyToSymbol(dcyp, &cyp,sizeof(Cylinder_Parameter));
        unsigned int grid_size1 = (p.N-1)/block_size + 1;
        unsigned int grid_size2 = (p.N*p.MN-1)/block_size + 1;
        unsigned int grid_size3 = (p.N_int - 1)/block_size + 1;
        unsigned int grid_size4 = (p.N_int*p.MN-1)/block_size + 1;
        unsigned int grid_size5 = (p.N_int-p.nx*p.ny)/block_size + 1;
        CHECK(cudaMemcpy(ba.x, x, p.N * sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(ba.y, y, p.N * sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(ba.z, z, p.N * sizeof(real), cudaMemcpyHostToDevice));
        long double start = cpuSecond();
        CHECK(cudaMemset(ba.NN,0,p.N* sizeof(int)));
        kernel_find_neighbor_3D<<<grid_size1,block_size>>>(ba.x, ba.y, ba.z, ba.NN,bb.NL);

        //int *disp_x = (int*) malloc(p.N* sizeof(int));
//        CHECK(cudaMemcpy(disp_x, ba.NN, p.N* sizeof(int), cudaMemcpyDeviceToHost));
//        cout<<"m "<<disp_x[0]<<endl;
        surface_correct_gpu(d, ba, bb);
        base_integrate_initial_gpu(d,ba, bb);
        real *disp_x = (real*) malloc(p.N*p.MN* sizeof(real));
                CHECK(cudaMemcpy(disp_x, bb.scr, p.N*p.MN*sizeof(real), cudaMemcpyDeviceToHost));
                for(int i=0; i<p.MN;i++)
                cout<<"m "<<disp_x[(0)*p.MN+i]<<endl;
        kernel_set_crack<<<grid_size4,block_size>>>(ba.NN, bb.NL, bb.fail, ba.x, ba.y);
        //kernel_vol_Corr<<<grid_size2, block_size>>>(ba.NN, bb.NL, ba.x, bb.idist, bb.fac, ba.y, ba.z);

//        kernel_initial_weight_3D<<<grid_size2, block_size,smem>>>(
//                ba.m, ba.NN, bb.w, bb.idist, bb.fac);
//        real *disp_x = (real*) malloc(p.N* sizeof(real));
//        CHECK(cudaMemcpy(disp_x, ba.m, p.N* sizeof(real), cudaMemcpyDeviceToHost));
//        cout<<"m "<<disp_x[0]<<endl;
        kernel_initial_cylinder<<<1,32>>>(cy.vel, cy.disp, cy.acc, cy.force);

        for (int i=0; i<p.nt; i++)
        {

            kernel_integrate_3D_1<<<grid_size3,block_size>>>(
                    ba.vel_x, ba.acc_x, ba.disp_x, ba.vel_y, ba.acc_y, ba.disp_y,
                    ba.vel_z, ba.acc_z, ba.disp_z);
            kernel_cy_integrate_1<<<1,32>>>(cy.vel, cy.disp, cy.acc);
            kernel_bond_force_3D<<<grid_size2,block_size,smem*3>>>(
                    ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y, ba.pforce_y,  ba.z, ba.disp_z, ba.pforce_z, ba.NN, bb.NL,
                    bb.idist, bb.scr, bb.fail, bb.fac);
//            kernel_cal_theta_3D<<<grid_size2, block_size, smem>>>(
//                    ba.NN, ba.m, ba.theta, bb.NL, bb.w, bb.fail, bb.e, bb.idist,bb.fac,
//                    ba.x, ba.disp_x, ba.y, ba.disp_y, ba.z, ba.disp_z);
//            kernel_state_force_3D<<<grid_size4, block_size, smem*3>>>(
//                    ba.NN, ba.m, ba.theta,  bb.fail, bb.NL, bb.e, bb.w, bb.idist, bb.fac,
//                    ba.x, ba.disp_x, ba.pforce_x, ba.y, ba.disp_y,ba.pforce_y, ba.z, ba.disp_z, ba.pforce_z);
            kernel_contact<<<grid_size1,block_size>>>(
                    loc_begin, ba.dmg,ba.x, ba.disp_x, ba.vel_x, ba.y, ba.disp_y, ba.vel_y,
                    ba.z, ba.disp_z, ba.vel_z, cy.force, cy.disp);
            kernel_integrate_3D_2<<<grid_size3,block_size>>>
                    (ba.vel_x, ba.acc_x, ba.pforce_x, ba.bforce_x, ba.vel_y, ba.acc_y, ba.pforce_y, ba.bforce_y,
                     ba.vel_z, ba.acc_z, ba.pforce_z, ba.bforce_z);
            kernel_cy_integrate_2<<<1,32>>>(cy.vel, cy.acc, cy.force);
            //cout<<i<<endl;

        }
        kernel_cal_dmg_3D<<<grid_size4,block_size, smem*2>>>(ba.NN,ba.dmg, bb.fail, bb.fac);
        CHECK(cudaDeviceSynchronize());
        cout<<"time "<<cpuSecond() - start<<endl;
        save_kw_gpu(ba);
    }

    Base_Free(ba,bb, d);
    cy_free(d,cy);
}