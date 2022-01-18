//
// Created by wxm on 2021/12/3.
//

#include "Integrate.cuh"

__global__ void kernel_cnn(
        real *cn_xy, real *pforce_x, real *pforceold_x,   real *velhalfold_x,
        real *disp_x, real *disp_xy, real *pforce_y, real *pforceold_y,
        real *velhalfold_y, real *disp_y, real *pforce_z, real *pforceold_z,
        real *velhalfold_z, real *disp_z
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    real cn = 0.0;
    real disp = 0.0;
    if (i<dp.N_int)
    {
        if (velhalfold_x[i] != 0.0)
            cn = -1.0 * disp_x[i]*disp_x[i] * (pforce_x[i]/dp.mass - pforceold_x[i]/dp.mass)/(dp.dt*velhalfold_x[i]);
        disp = disp_x[i]*disp_x[i];
        if (disp_y != NULL){
            if (velhalfold_y[i] != 0.0)
                cn -= 1.0 * disp_y[i]*disp_y[i] * (pforce_y[i]/dp.mass - pforceold_y[i]/dp.mass)/(dp.dt*velhalfold_y[i]);
            disp += disp_y[i] *disp_y[i];
            if (disp_z != NULL){
                if (velhalfold_z[i] != 0.0)
                    cn -= 1.0 * disp_z[i]*disp_z[i] * (pforce_z[i]/dp.mass - pforceold_z[i]/dp.mass)/(dp.dt*velhalfold_z[i]);
                disp += disp_z[i] *disp_z[i];
            }
        }
        cn_xy[i] = cn;
        disp_xy[i] = disp;
    }
    //   }
//    cn_xy[i] = 0.0;
//    disp_xy[i] = 0.0;
//    if (i<dp.N_int)
//    {
//        if (velhalfold_x[i] != 0.0)
//            cn_xy[i] -= 1.0 * gsquare(disp_x[i]) * ((pforce_x[i]- pforceold_x[i])/dp.mass )/(dp.dt*velhalfold_x[i]);
//
//        if (velhalfold_y[i] != 0.0)
//            cn_xy[i] -= 1.0 * gsquare(disp_y[i]) * ((pforce_y[i] - pforceold_y[i])/dp.mass )/(dp.dt*velhalfold_y[i]);
//        disp_xy[i] += disp_y[i] *disp_y[i] + disp_x[i]*disp_x[i];;
//        //if(i==0) printf("cn %e disp %e\n", cn_xy[i],disp_xy[i]);

    //}
}




__global__ void kernel_integrate(
        int ct, real cn,  real *pforce_x, real *velhalf_x, real *bforce_x,
        real *vel_x,  real *velhalfold_x,  real *disp_x, real *pforceold_x, real *pforce_y,
        real *velhalf_y, real *bforce_y,  real *vel_y, real *velhalfold_y,
        real *disp_y,  real *pforceold_y, real *pforce_z, real *velhalf_z, real *bforce_z,
        real *vel_z, real *velhalfold_z, real *disp_z,   real *pforceold_z
)
{
    int	i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<dp.N_int)
    {
        if(ct == 0)
        {
            velhalf_x[i] = 1.0 * dp.dt / dp.mass * (pforce_x[i] + bforce_x[i]) / 2.0;
            if (vel_y!=NULL)
            {
                velhalf_y[i] = 1.0 * dp.dt / dp.mass * (pforce_y[i] + bforce_y[i]) / 2.0;
                if (vel_z!=NULL)
                {
                    velhalf_z[i] = 1.0 * dp.dt / dp.mass * (pforce_z[i] + bforce_z[i]) / 2.0;
                }
            }
        }
        else
        {
            velhalf_x[i] = ((2.0 - cn * dp.dt) * velhalfold_x[i] + 2.0 * dp.dt / dp.mass * (pforce_x[i] + bforce_x[i])) / (2.0 + cn * dp.dt);
            if (vel_y!=NULL)
            {
                velhalf_y[i] = ((2.0 - cn * dp.dt) * velhalfold_y[i] + 2.0 * dp.dt / dp.mass * (pforce_y[i] + bforce_y[i])) / (2.0 + cn * dp.dt);
                if (vel_z!=NULL)
                    velhalf_z[i] = ((2.0 - cn * dp.dt) * velhalfold_z[i] + 2.0 * dp.dt / dp.mass * (pforce_z[i] + bforce_z[i])) / (2.0 + cn * dp.dt);
            }
        }
        vel_x[i] = 0.5 * (velhalfold_x[i] + velhalf_x[i]);
        disp_x[i] += velhalf_x[i] * dp.dt;
        velhalfold_x[i] = velhalf_x[i];
        pforceold_x[i] = pforce_x[i];
        if (vel_y!=NULL)
        {
            vel_y[i] = 0.5 * (velhalfold_y[i] + velhalf_y[i]);
            disp_y[i] += velhalf_y[i] * dp.dt;
            velhalfold_y[i] = velhalf_y[i];
            pforceold_y[i] = pforce_y[i];
            if (vel_z!=NULL)
            {
                vel_z[i] = 0.5 * (velhalfold_z[i] + velhalf_z[i]);
                disp_z[i] += velhalf_z[i] * dp.dt;
                velhalfold_z[i] = velhalf_z[i];
                pforceold_z[i] = pforce_z[i];
            }
        }
    }
}

real cal_cn( BAtom &ba)
{
    real cn = 0.0;

    real cn1 = sum(p.N_int, ba.cn_xy);
    real cn2 = sum(p.N_int, ba.disp_xy);
    //real cn1 = thrust::reduce(thrust::device, ba.cn_xy, ba.cn_xy+p.N);
    //real cn2 = thrust::reduce(thrust::device, ba.disp_xy, ba.disp_xy+p.N);;
    //cout<<"cn1 "<<cn1<<" cn2 "<<cn2<<endl;
    if(cn2 != 0.0)
    {
        if (cn1/cn2 > 0.0)
            cn = 2.0 * sqrt(cn1/cn2);
        else
            cn = 0.0;
    }
    else
        cn = 0.0;
    if (cn>2.0)
        cn = 1.9;
    return cn;
}
void static_integrate_gpu(Dselect &d, BAtom &ba, int ct)
{
    int grid_size = (p.N_int-1)/block_size + 1;
    if (d.Dim==1)
    {
        kernel_cnn<<<grid_size, block_size>>>(
                ba.cn_xy,ba.pforce_x, ba.pforceold_x,  ba.velhalfold_x,
                ba.disp_x, ba.disp_xy);
        real cn = cal_cn(ba);
        kernel_integrate<<<grid_size, block_size>>>(
                ct,cn,   ba.pforce_x, ba.velhalf_x, ba.bforce_x,
                ba.vel_x,  ba.velhalfold_x,  ba.disp_x, ba.pforceold_x);
    } else if (d.Dim==2){
        kernel_cnn<<<grid_size, block_size>>>(
                ba.cn_xy,ba.pforce_x, ba.pforceold_x,  ba.velhalfold_x,
                ba.disp_x, ba.disp_xy, ba.pforce_y, ba.pforceold_y,
                ba.velhalfold_y, ba.disp_y);

        real cn = cal_cn(ba);
        kernel_integrate<<<grid_size, block_size>>>(
                ct, cn,  ba.pforce_x, ba.velhalf_x, ba.bforce_x,
                ba.vel_x,  ba.velhalfold_x,  ba.disp_x, ba.pforceold_x, ba.pforce_y,
                ba.velhalf_y, ba.bforce_y,  ba.vel_y, ba.velhalfold_y,
                ba.disp_y,  ba.pforceold_y);

    } else if (d.Dim==3){

        kernel_cnn<<<grid_size, block_size>>>(
                ba.cn_xy,ba.pforce_x, ba.pforceold_x,  ba.velhalfold_x,
                ba.disp_x, ba.disp_xy, ba.pforce_y, ba.pforceold_y,
                ba.velhalfold_y, ba.disp_y, ba.pforce_z, ba.pforceold_z,
                ba.velhalfold_z, ba.disp_z);
        real cn = cal_cn(ba);
        kernel_integrate<<<grid_size, block_size>>>(
                ct,cn,   ba.pforce_x, ba.velhalf_x, ba.bforce_x,
                ba.vel_x,  ba.velhalfold_x,  ba.disp_x, ba.pforceold_x, ba.pforce_y,
                ba.velhalf_y, ba.bforce_y, ba.vel_y, ba.velhalfold_y,
                ba.disp_y,  ba.pforceold_y, ba.pforce_z,
                ba.velhalf_z, ba.bforce_z,  ba.vel_z, ba.velhalfold_z,
                ba.disp_z,  ba.pforceold_z);
    }

}

__global__ void kernel_integrate_2D_1(
        real *vel_x, real *acc_x,  real *disp_x, real *vel_y, real *acc_y, real *disp_y)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<dp.N)
    {
        vel_x[i] += (dp.dt/2.0)*acc_x[i];
        vel_y[i] += (dp.dt/2.0)*acc_y[i];
        disp_x[i] += vel_x[i]*dp.dt;
        disp_y[i] += vel_y[i]*dp.dt;
    }
}


__global__ void  kernel_integrate_3D_1(
        real *vel_x, real *acc_x,  real *disp_x, real *vel_y, real *acc_y, real *disp_y,
        real *disp_z, real *vel_z, real *acc_z)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<dp.N)
    {
        vel_x[i] += (dp.dt/2.0)*acc_x[i];
        vel_y[i] += (dp.dt/2.0)*acc_y[i];
        vel_z[i] += (dp.dt/2.0)*acc_z[i];
        disp_x[i] += vel_x[i]*dp.dt;
        disp_y[i] += vel_y[i]*dp.dt;
        disp_z[i] += vel_z[i]*dp.dt;
    }
}

inline void gpu_central_difference2D_1(BAtom &ba)
{
    int grid_szie = (p.N_int-1)/block_size + 1;
    kernel_integrate_2D_1<<<grid_szie,block_size>>>(
            ba.vel_x, ba.acc_x, ba.disp_x, ba.vel_y, ba.acc_y, ba.disp_y);
}

inline void gpu_central_difference3D_1(BAtom &ba)
{
    int grid_szie = (p.N_int-1)/block_size + 1;
    kernel_integrate_3D_1<<<grid_szie,block_size>>>(
            ba.vel_x, ba.acc_x, ba.disp_x, ba.vel_y, ba.acc_y, ba.disp_y, ba.vel_z, ba.acc_z, ba.disp_z);
}

__global__ void  kernel_integrate_2D_2(
        real *vel_x, real *acc_x, real *pforce_x, real *bforce_x,
        real *vel_y, real *acc_y,  real *pforce_y, real *bforce_y)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<dp.N)
    {
        //if(i==N-500) printf("moca %e %e\n",bforce_x[i],pforce_y[i]);
        acc_x[i] = (pforce_x[i] + bforce_x[i])/dp.dens;
        acc_y[i] = (pforce_y[i] + bforce_y[i])/dp.dens;
        vel_x[i] += (dp.dt/2.0)*acc_x[i];
        vel_y[i] += (dp.dt/2.0)*acc_y[i];
    }
}

__global__ void  kernel_integrate_3D_2(
        real *vel_x, real *acc_x, real *pforce_x, real *bforce_x,
        real *vel_y, real *acc_y,  real *pforce_y, real *bforce_y,
        real *vel_z, real *acc_z, real *pforce_z, real *bforce_z)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<dp.N)
    {

        acc_x[i] = (pforce_x[i] + bforce_x[i])/dp.dens;
        acc_y[i] = (pforce_y[i] + bforce_y[i])/dp.dens;
        acc_z[i] = (pforce_z[i] + bforce_z[i])/dp.dens;
        vel_x[i] += (dp.dt/2.0)*acc_x[i];
        vel_y[i] += (dp.dt/2.0)*acc_y[i];
        vel_z[i] += (dp.dt/2.0)*acc_z[i];
    }
}

inline void central_difference_2D_2_gpu(BAtom &ba)
{
    int grid_szie = (p.N_int-1)/block_size + 1;
    kernel_integrate_2D_2<<<grid_szie,block_size>>>
            (ba.vel_x, ba.acc_x, ba.pforce_x, ba.bforce_x, ba.vel_y, ba.acc_y, ba.pforce_y, ba.bforce_y);

}

inline void central_difference_3D_2_gpu(BAtom &ba)
{
    int grid_szie = (p.N_int-1)/block_size + 1;
    kernel_integrate_3D_2<<<grid_szie,block_size>>>(ba.vel_x, ba.acc_x, ba.pforce_x, ba.bforce_x,
                                                    ba.vel_y, ba.acc_y, ba.pforce_y, ba.bforce_y,
                                                    ba.vel_z, ba.acc_z, ba.pforce_z, ba.bforce_z
    );
}


real  cal_cn_cpu(real *pforce_x,   real *velhalfold_x,  real *disp_x,
                 real *pforceold_x, real *pforce_y, real *velhalfold_y,
                 real *disp_y,  real *pforceold_y, real *pforce_z,
                 real *velhalfold_z, real *disp_z,   real *pforceold_z)
{
    real cn = 0.0;
    real cn1 = 0.0;
    real cn2 = 0.0;
    for (int i=0; i<p.N_int; i++)
    {
        if (velhalfold_x[i] != 0.0)
            cn1 -= 1.0 * disp_x[i]*disp_x[i] * (pforce_x[i]/p.mass - pforceold_x[i]/p.mass)/(p.dt*velhalfold_x[i]);
        cn2 += disp_x[i]*disp_x[i];
        if (disp_y != NULL){
            if (velhalfold_y[i] != 0.0)
                cn1 -= 1.0 * disp_y[i]*disp_y[i] * (pforce_y[i]/p.mass - pforceold_y[i]/p.mass)/(p.dt*velhalfold_y[i]);
            cn2 += disp_y[i] *disp_y[i];
            if (disp_z != NULL){
                if (velhalfold_z[i] != 0.0)
                    cn1 -= 1.0 * disp_z[i]*disp_z[i] * (pforce_z[i]/p.mass - pforceold_z[i]/p.mass)/(p.dt*velhalfold_z[i]);
                cn2 += disp_z[i] *disp_z[i];
            }
        }
        //if (i==0) cout<<"cn "<<cn1<<" cn "<<cn2<<endl;
    }
    //cout<<"cn1 "<<cn1<<" cn2 "<<cn2<<endl;
    if(cn2 != 0.0)
    {
        if (cn1/cn2 > 0.0)
            cn = 2.0 * sqrt(cn1/cn2);
        else
            cn = 0.0;
    }
    else
        cn = 0.0;
    if (cn>2.0)
        cn = 1.9;
    return cn;
}

void cal_cn3_cpu(real *cn_xy,real *disp_xy, real *pforce_x,   real *velhalfold_x,  real *disp_x,
                 real *pforceold_x, real *pforce_y, real *velhalfold_y,
                 real *disp_y,  real *pforceold_y, real *pforce_z,
                 real *velhalfold_z, real *disp_z,   real *pforceold_z)
{
    real cn = 0.0;

    for (int i=0; i<p.N_int; i++)
    {
        cn_xy[i] = 0.0;
        disp_xy[i] = 0.0;
        if (velhalfold_x[i] != 0.0)
            cn_xy[i] = -1.0 * disp_x[i]*disp_x[i] * (pforce_x[i]/p.mass - pforceold_x[i]/p.mass)/(p.dt*velhalfold_x[i]);
        disp_xy[i] = disp_x[i]*disp_x[i];
        if (disp_y != NULL){
            if (velhalfold_y[i] != 0.0)
                cn_xy[i] -= 1.0 * disp_y[i]*disp_y[i] * (pforce_y[i]/p.mass - pforceold_y[i]/p.mass)/(p.dt*velhalfold_y[i]);
            disp_xy[i] += disp_y[i] *disp_y[i];
            if (disp_z != NULL){
                if (velhalfold_z[i] != 0.0)
                    cn_xy[i] -= 1.0 * disp_z[i]*disp_z[i] * (pforce_z[i]/p.mass - pforceold_z[i]/p.mass)/(p.dt*velhalfold_z[i]);
                disp_xy[i] += disp_z[i] *disp_z[i];
            }
        }
    }
}

real cal_cn2_cpu(real *cn_xy, real *disp_xy)

{
    real cn = 0.0;
    real cn1 = 0.0;
    real cn2 = 0.0;
    for (int i=0; i<p.N_int; i++)
    {
        cn1+=cn_xy[i];
        cn2+=disp_xy[i];
    }
    cout<<"cn1 "<<cn1<<" cn2 "<<cn2<<endl;
    if(cn2 != 0.0)
    {
        if (cn1/cn2 > 0.0)
            cn = 2.0 * sqrt(cn1/cn2);
        else
            cn = 0.0;
    }
    else
        cn = 0.0;
    if (cn>2.0)
        cn = 1.9;
    return cn;
}

void static_integrate_cpu(
        int ct, real *pforce_x, real *velhalf_x, real *bforce_x, real *vel_x,  real *velhalfold_x,  real *disp_x,
        real *pforceold_x, real *pforce_y, real *velhalf_y, real *bforce_y,  real *vel_y, real *velhalfold_y,
        real *disp_y,  real *pforceold_y, real *pforce_z, real *velhalf_z, real *bforce_z,
        real *vel_z, real *velhalfold_z, real *disp_z,   real *pforceold_z)
{
    //计算阻尼
    real cn = 0.0;
    real cn1 = 0.0;
    real cn2 = 0.0;
    for (int i=0; i<p.N_int; i++)
    {
        if (velhalfold_x[i] != 0.0)
            cn1 = -1.0 * disp_x[i]*disp_x[i] * (pforce_x[i]/p.mass - pforceold_x[i]/p.mass)/(p.dt*velhalfold_x[i]);
        cn2 = disp_x[i]*disp_x[i];
        if (disp_y != NULL){
            if (velhalfold_y[i] != 0.0)
                cn1 -= 1.0 * disp_y[i]*disp_y[i] * (pforce_y[i]/p.mass - pforceold_y[i]/p.mass)/(p.dt*velhalfold_y[i]);
            cn2 += disp_y[i] *disp_y[i];
            if (disp_z != NULL){
                if (velhalfold_z[i] != 0.0)
                    cn1 -= 1.0 * disp_z[i]*disp_z[i] * (pforce_z[i]/p.mass - pforceold_z[i]/p.mass)/(p.dt*velhalfold_z[i]);
                cn2 += disp_z[i] *disp_z[i];
            }
        }
    }

    if(cn2 != 0.0)
    {
        if (cn1/cn2 > 0.0)
            cn = 2.0 * sqrt(cn1/cn2);
        else
            cn = 0.0;
    }
    else
        cn = 0.0;
    if (cn>2.0)
        cn = 1.9;

    //积分
    for (int i=0; i<p.N_int; i++)
    {
        if(ct == 0)
        {
            velhalf_x[i] = 1.0 * p.dt / p.mass * (pforce_x[i] + bforce_x[i]) / 2.0;
            if (vel_y!=NULL)
            {
                velhalf_y[i] = 1.0 * p.dt / p.mass * (pforce_y[i] + bforce_y[i]) / 2.0;
                if (vel_z!=NULL)
                {
                    velhalf_z[i] = 1.0 * p.dt / p.mass * (pforce_z[i] + bforce_z[i]) / 2.0;
                }
            }
        }
        else
        {
            velhalf_x[i] = ((2.0 - cn * p.dt) * velhalfold_x[i] + 2.0 * p.dt / p.mass * (pforce_x[i] + bforce_x[i])) / (2.0 + cn * p.dt);
            if (vel_y!=NULL)
            {
                velhalf_y[i] = ((2.0 - cn * p.dt) * velhalfold_y[i] + 2.0 * p.dt / p.mass * (pforce_y[i] + bforce_y[i])) / (2.0 + cn * p.dt);
                if (vel_z!=NULL)
                    velhalf_z[i] = ((2.0 - cn * p.dt) * velhalfold_z[i] + 2.0 * p.dt / p.mass * (pforce_z[i] + bforce_z[i])) / (2.0 + cn * p.dt);
            }
        }
        vel_x[i] = 0.5 * (velhalfold_x[i] + velhalf_x[i]);
        disp_x[i] += velhalf_x[i] * p.dt;
        velhalfold_x[i] = velhalf_x[i];
        pforceold_x[i] = pforce_x[i];
        if (vel_y!=NULL)
        {
            vel_y[i] = 0.5 * (velhalfold_y[i] + velhalf_y[i]);
            disp_y[i] += velhalf_y[i] * p.dt;
            velhalfold_y[i] = velhalf_y[i];
            pforceold_y[i] = pforce_y[i];
            if (vel_z!=NULL)
            {
                vel_z[i] = 0.5 * (velhalfold_z[i] + velhalf_z[i]);
                disp_z[i] += velhalf_z[i] * p.dt;
                velhalfold_z[i] = velhalf_z[i];
                pforceold_z[i] = pforce_z[i];
            }
        }
    }
}

void integrate_1(real *vel_x, real *acc_x,  real *disp_x, real *vel_y, real *acc_y, real *disp_y,
                 real *vel_z, real *acc_z,  real *disp_z)
{
    for (int i=0; i<p.N_int; i++)
    {
        vel_x[i] += (p.dt/2.0)*acc_x[i];
        vel_y[i] += (p.dt/2.0)*acc_y[i];
        disp_x[i] += vel_x[i]*p.dt;
        disp_y[i] += vel_y[i]*p.dt;

        if(disp_z!=NULL)
        {
            vel_z[i] += (p.dt/2.0)*acc_z[i];
            disp_z[i] += vel_z[i]*p.dt;
        }
    }
}

void integrate_2(real *vel_x, real *acc_x, real *pforce_x, real *bforce_x,
                        real *vel_y, real *acc_y, real *pforce_y, real *bforce_y,
                        real *vel_z, real *acc_z, real *pforce_z, real *bforce_z)
{
    for (int i=0; i<p.N_int; i++)
    {
        acc_x[i] = (pforce_x[i] + bforce_x[i])/p.dens;
        acc_y[i] = (pforce_y[i] + bforce_y[i])/p.dens;
        vel_x[i] += (p.dt/2.0)*acc_x[i];
        vel_y[i] += (p.dt/2.0)*acc_y[i];
        if(vel_z!=NULL)
        {
            acc_z[i] = (pforce_z[i] + bforce_z[i])/p.dens;
            vel_z[i] += (p.dt/2.0)*acc_z[i];
        }
    }
}

