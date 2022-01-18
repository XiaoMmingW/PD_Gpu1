//
// Created by wxm on 2021/12/2.
//

#include "Memory.cuh"
static void base_alloc_memory_gpu(
        const Dselect &d,void **coord, void **disp, void **vel, void **acc, void **bforce,
        void **pforce, void **pforceold, void **velhalf, void **velhalfold)
{
    size_t real_m1 = p.N*sizeof(real);

    CHECK(cudaMalloc(coord, real_m1));
    CHECK(cudaMalloc(disp, real_m1));
    CHECK(cudaMalloc(vel, real_m1));
    CHECK(cudaMalloc(acc, real_m1));
    CHECK(cudaMalloc(bforce, real_m1));
    CHECK(cudaMalloc(pforce, real_m1));
    if (d.Mode==1)
    {
        CHECK(cudaMalloc(pforceold, real_m1));
        CHECK(cudaMalloc(velhalf, real_m1));
        CHECK(cudaMalloc(velhalfold, real_m1));
    }
}




void Base_Allocate_gpu( BAtom &ba, BBond &bb, const Dselect &d) {

    size_t real_m1 = p.N*sizeof(real);
    size_t real_mn = p.N * p.MN * sizeof(real);
    size_t int_m1 = p.N * sizeof(int);
    size_t int_mn = p.N * p.MN * sizeof(int);

    CHECK(cudaMalloc((void**)&ba.dmg,  p.N_int*sizeof(real)));
    CHECK(cudaMalloc((void**)&ba.NN, int_m1));
    CHECK(cudaMalloc((void**)&bb.NL, int_mn));
    CHECK(cudaMalloc((void**)&bb.fail, p.N_int*p.MN*sizeof(int)));

    CHECK(cudaMalloc((void**)&bb.idist, real_mn));
    CHECK(cudaMalloc((void**)&bb.scr, real_mn));
    CHECK(cudaMalloc((void**)&bb.fac, real_mn));

    if(d.B_S==2)
    {
        CHECK(cudaMalloc((void**)&bb.nlength, real_mn));

        CHECK(cudaMalloc((void**)&bb.bond_length, real_mn)); // ?
        CHECK(cudaMalloc((void**)&ba.m, real_m1));
        CHECK(cudaMalloc((void**)&ba.theta, real_m1));
        CHECK(cudaMalloc((void**)&bb.w, real_mn));
        CHECK(cudaMalloc((void**)&bb.e, real_mn));
    }
    base_alloc_memory_gpu(d, (void**)& ba.x,  (void**)& ba.disp_x, (void**)& ba.vel_x, (void**)& ba.acc_x,
                          (void**)& ba.bforce_x, (void**)& ba.pforce_x,  (void**)&ba.pforceold_x,
                          (void**)&ba.velhalf_x, (void**)&ba.velhalfold_x);
    if(d.Dim==2 |d.Dim==3)
    {
        base_alloc_memory_gpu(d, (void**)& ba.y, (void**)& ba.disp_y, (void**)& ba.vel_y, (void**)& ba.acc_y,
                              (void**)& ba.bforce_y, (void**)& ba.pforce_y, (void**)&ba.pforceold_y,
                              (void**)&ba.velhalf_y, (void**)&ba.velhalfold_y);
        if (d.Dim==3)
        {
            base_alloc_memory_gpu(d, (void**)& ba.z, (void**)& ba.disp_z, (void**)& ba.vel_z, (void**)& ba.acc_z,
                                  (void**)& ba.bforce_z, (void**)& ba.pforce_z,(void**)&ba.pforceold_z,
                                  (void**)&ba.velhalf_z, (void**)&ba.velhalfold_z);
        }
    }
    if (d.Mode == 1)
    {

        CHECK(cudaMalloc((void**)& ba.cn_xy, real_m1));
        CHECK(cudaMalloc((void**)& ba.disp_xy, real_m1));
    }
}

static void base_alloc_memory_cpu(
        const Dselect &d,real **coord, real **disp, real **vel, real **acc, real **bforce, real **pforce,
        real **pforceold, real **velhalf, real **velhalfold)
{
    size_t real_m1 = p.N*sizeof(real);

    *coord = (real*) malloc(real_m1);
    *disp = (real*) malloc(real_m1);
    *vel = (real*) malloc(real_m1);
    *acc = (real*) malloc(real_m1);
    *bforce = (real*) malloc(real_m1);
    *pforce = (real*) malloc(real_m1);

    if (d.Mode==1)
    {
        *pforceold = (real*) malloc(real_m1);
        *velhalf = (real*) malloc(real_m1);
        *velhalfold= (real*) malloc(real_m1);
    }
}

void Base_Allocate_cpu( BAtom &ba, BBond &bb, const Dselect &d) {

    size_t real_m1 = p.N*sizeof(real);
    size_t real_mn = p.N * p.MN * sizeof(real);
    size_t int_m1 = p.N * sizeof(int);
    size_t int_mn = p.N * p.MN * sizeof(int);
    ba.dmg = (real *) malloc(p.N_int* sizeof(real));
    ba.NN = (int *) malloc(int_m1);
    bb.NL = (int *) malloc(int_mn);
    bb.fail = (int *) malloc(p.N_int*p.MN* sizeof(int));
    bb.idist = (real *)malloc(real_mn);
    bb.scr = (real *)malloc(real_mn);
    bb.fac = (real *)malloc(real_mn);

    if(d.B_S==2)
    {
        ba.m = (real *)malloc(real_m1);
        ba.theta = (real *)malloc(real_m1);
        bb.nlength = (real *)malloc(real_mn);
        bb.bond_length = (real *)malloc(real_mn);
        bb.w = (real *)malloc(real_mn);
        bb.e = (real *)malloc(real_mn);
    }
    base_alloc_memory_cpu(d, & ba.x, & ba.disp_x, & ba.vel_x, & ba.acc_x,
                          & ba.bforce_x, & ba.pforce_x,  &ba.pforceold_x,
                          &ba.velhalf_x, &ba.velhalfold_x);
    if(d.Dim==2 |d.Dim==3)
    {
        base_alloc_memory_cpu(d, & ba.y,& ba.disp_y, & ba.vel_y, & ba.acc_y,
                              & ba.bforce_y, & ba.pforce_y, &ba.pforceold_y,
                              &ba.velhalf_y, &ba.velhalfold_y);
        if (d.Dim==3)
        {
            base_alloc_memory_cpu(d, &ba.z, &ba.disp_z, &ba.vel_z, &ba.acc_z,
                                  & ba.bforce_z, & ba.pforce_z, &ba.pforceold_z,
                                  &ba.velhalf_z, &ba.velhalfold_z);
        }
    }
}

void Base_Allocate( BAtom &ba, BBond &bb, const Dselect &d)
{
    if (d.device==1)
    {
        Base_Allocate_cpu( ba, bb, d);
    } else {
        Base_Allocate_gpu( ba, bb, d);
    }
}

static void base_free_memory_gpu(
        const Dselect &d,real *coord, real *disp, real *vel, real *acc,  real *bforce,
        real *pforce,  real *pforceold, real *velhalf, real *velhalfold)
{
    CHECK(cudaFree(coord));
    CHECK(cudaFree(disp));
    CHECK(cudaFree(vel));
    CHECK(cudaFree(acc));
    CHECK(cudaFree(bforce));
    CHECK(cudaFree(pforce));

    if (d.Mode==1)
    {
        CHECK(cudaFree(pforceold));
        CHECK(cudaFree(velhalf));
        CHECK(cudaFree(velhalfold));
    }
}

static void base_free_memory_cpu(
        const Dselect &d,real *coord, real *disp, real *vel, real *acc,  real *bforce,
        real *pforce,  real *pforceold, real *velhalf, real *velhalfold)
{
    CHECK(cudaFree(coord));
    CHECK(cudaFree(disp));
    CHECK(cudaFree(vel));
    CHECK(cudaFree(acc));
    CHECK(cudaFree(bforce));
    CHECK(cudaFree(pforce));
    free(coord);
    free(disp);
    free(vel);
    free(vel);
    free(acc);
    free(bforce);
    free(pforce);
    if (d.Mode==1)
    {
        free(pforceold);
        free(velhalf);
        free(velhalfold);
    }
}

void base_free_gpu(BAtom &ba, BBond &bb, const Dselect &d)
{
    CHECK(cudaFree(ba.dmg));
    CHECK(cudaFree(ba.NN));
    CHECK(cudaFree(bb.NL));
    CHECK(cudaFree(bb.fail));

    CHECK(cudaFree(bb.idist));
    CHECK(cudaFree(bb.scr));
    CHECK(cudaFree(bb.fac));

    if(d.B_S==2)
    {
        CHECK(cudaFree(bb.nlength));

        CHECK(cudaFree(bb.bond_length)); // ?
        CHECK(cudaFree(ba.m));
        CHECK(cudaFree(ba.theta));
        CHECK(cudaFree(bb.w));
        CHECK(cudaFree(bb.e));
    }
    base_free_memory_gpu(d, ba.x,   ba.disp_x,  ba.vel_x,  ba.acc_x,
                         ba.bforce_x,  ba.pforce_x,  ba.pforceold_x,
                         ba.velhalf_x, ba.velhalfold_x);
    if(d.Dim==2 |d.Dim==3)
    {
        base_free_memory_gpu(d,  ba.y,  ba.disp_y,  ba.vel_y,  ba.acc_y,
                             ba.bforce_y,  ba.pforce_y, ba.pforceold_y,
                             ba.velhalf_y, ba.velhalfold_y);
        if (d.Dim==3)
        {
            base_free_memory_gpu(d,  ba.z,  ba.disp_z,  ba.vel_z,  ba.acc_z,
                                 ba.bforce_z,  ba.pforce_z,ba.pforceold_z,
                                 ba.velhalf_z, ba.velhalfold_z);
        }
    }
    if (d.Mode == 1)
    {

        CHECK(cudaFree(ba.cn_xy));
        CHECK(cudaFree(ba.disp_xy));
    }
}

void base_free_cpu(BAtom &ba, BBond &bb, const Dselect &d)
{
    free(ba.dmg);
    free(ba.NN);
    free(bb.NL);
    free(bb.fail);

    free(bb.idist);
    free(bb.scr);
    free(bb.fac);

    if(d.B_S==2)
    {
        free(bb.nlength);

        free(bb.bond_length); // ?
        free(ba.m);
        free(ba.theta);
        free(bb.w);
        free(bb.e);
    }
    base_free_memory_cpu(d, ba.x,   ba.disp_x,  ba.vel_x,  ba.acc_x,
                         ba.bforce_x,  ba.pforce_x,  ba.pforceold_x,
                         ba.velhalf_x, ba.velhalfold_x);
    if(d.Dim==2 |d.Dim==3)
    {
        base_free_memory_cpu(d,  ba.y,  ba.disp_y,  ba.vel_y,  ba.acc_y,
                             ba.bforce_y,  ba.pforce_y, ba.pforceold_y,
                             ba.velhalf_y, ba.velhalfold_y);
        if (d.Dim==3)
        {
            base_free_memory_cpu(d,  ba.z,  ba.disp_z,  ba.vel_z,  ba.acc_z,
                                 ba.bforce_z,  ba.pforce_z,ba.pforceold_z,
                                 ba.velhalf_z, ba.velhalfold_z);
        }
    }

}

void Base_Free(BAtom &ba, BBond &bb, const Dselect &d)
{
    if (d.device==1)
    {
        Base_Allocate_cpu( ba, bb, d);
    } else {
        Base_Allocate_gpu( ba, bb, d);
    }
}

void cy_allocate(const Dselect &d, Cylinder &cy)
{
    if(d.device==1)
    {
        cy.acc = (real *) malloc(d.Dim* sizeof(real));
        cy.vel = (real *) malloc(d.Dim* sizeof(real));
        cy.disp = (real *) malloc(d.Dim* sizeof(real));
        cy.force = (real *) malloc(d.Dim* sizeof(real));
    } else {
        CHECK(cudaMalloc((void**)&cy.acc, d.Dim* sizeof(real)));
        CHECK(cudaMalloc((void**)&cy.disp, d.Dim* sizeof(real)));
        CHECK(cudaMalloc((void**)&cy.vel, d.Dim* sizeof(real)));
        CHECK(cudaMalloc((void**)&cy.force, d.Dim* sizeof(real)));
    }
}

void cy_free(const Dselect &d, Cylinder &cy)
{
    if(d.device==1)
    {
        free(cy.acc);
        free(cy.vel);
        free(cy.disp);
        free(cy.force);
    } else {
        CHECK(cudaFree(cy.acc));
        CHECK(cudaFree(cy.vel));
        CHECK(cudaFree(cy.disp));
        CHECK(cudaFree(cy.force));
    }
}