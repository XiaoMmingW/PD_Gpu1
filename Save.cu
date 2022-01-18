//
// Created by wxm on 2021/12/3.
//

#include "Save.cuh"
void save_disp_gpu(Dselect &d, BAtom &ba, const string FILE)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);
    size_t byte = p.N_int*sizeof(real);

    real *disp_x = (real*) malloc(byte);
    real *disp_y = (real*) malloc(byte);;
    real *dmg = (real*) malloc(byte);;
    real *disp_z = NULL;

    CHECK(cudaMemcpy(disp_x, ba.disp_x, byte, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(disp_y, ba.disp_y, byte, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dmg, ba.dmg, byte, cudaMemcpyDeviceToHost));
    if (d.Dim==3){
        disp_z = (real*) malloc(byte);;
        CHECK(cudaMemcpy(disp_z, ba.disp_z, byte, cudaMemcpyDeviceToHost));
    }
    if (d.Dim==2)
    {
        for(int i=0;i<p.N_int;i++)
            ofs<<disp_x[i]<<" "<<disp_y[i]<<" "<<dmg[i]<<endl;
    } else{
        for(int i=0;i<p.N_int;i++)
            ofs<<disp_x[i]<<" "<<disp_y[i]<<" "<<disp_z[i]<<" "<<dmg[i]<<endl;
    }
    free(disp_x);
    free(disp_y);
    free(dmg);
    if (d.Dim==3)
        free(disp_z);
    ofs.close();
}

void save_disp_cpu(Dselect &d, BAtom &ba, const string FILE)
{

    ofstream ofs;
    ofs.open(FILE, ios::out);
    size_t byte = p.N_int*sizeof(real);

    if (d.Dim==2)
    {
        for(int i=0;i<p.N_int;i++)
            ofs<<ba.disp_x[i]<<" "<<ba.disp_y[i]<<" "<<ba.dmg[i]<<endl;
    } else{
        for(int i=0;i<p.N_int;i++)
            ofs<<ba.disp_x[i]<<" "<<ba.disp_y[i]<<" "<<ba.disp_z[i]<<" "<<ba.dmg[i]<<endl;
    }
    ofs.close();
}

void save_kw_gpu( BAtom &ba, const string FILE)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);
    size_t byte = p.N_int*sizeof(real);

    real *disp_x = (real*) malloc(byte);
    real *disp_y = (real*) malloc(byte);
    real *disp_z = (real*) malloc(byte);
    real *dmg = (real*) malloc(byte);

    CHECK(cudaMemcpy(disp_x, ba.disp_x, byte, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(disp_y, ba.disp_y, byte, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(disp_z, ba.disp_z, byte, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dmg, ba.dmg, byte, cudaMemcpyDeviceToHost));
    int z_num = p.nz/2*0;
    cout<<z_num<<endl;
    unsigned int num = p.N_int/p.nz*z_num;
    cout<<"num "<<num<<endl;
    real tx, ty;
    bool flag;
    for(int j=0; j<p.ny;j++)
    {
        for(int i=0; i<p.nx; i++)
        {
            tx = (-(p.nx-1)/2.0 + i) * p.dx;
            ty = (-(p.ny-1)/2.0 + j) * p.dx;

            flag = (tx>(-cyp.rad-1.1*p.dx) && tx<(-cyp.rad+0.1*p.dx)) | (tx>(cyp.rad-0.1*p.dx) && tx<(cyp.rad+1.1*p.dx));
            if (flag)
            {
                if (ty>=-1.0e-10)
                    ofs<<NAN<<" "<<NAN<<" "<<0.0<<" "<<NAN<<endl;
                else
                {
                    ofs<<disp_x[num]<<" "<<disp_y[num]<<" "<<disp_z[num]<<" "<<dmg[num]<<endl;
                    num++;
                }
            } else {
                ofs<<disp_x[num]<<" "<<disp_y[num]<<" "<<disp_z[num]<<" "<<dmg[num]<<endl;
                num++;
            }
        }
    }
}

void save_kw_cpu( BAtom &ba, const string FILE)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);
    size_t byte = p.N_int*sizeof(real);


    int z_num = p.nz/2*0;
    cout<<z_num<<endl;
    unsigned int num = p.N_int/p.nz*z_num;
    cout<<"num "<<num<<endl;
    real tx, ty;
    bool flag;
    for(int j=0; j<p.ny;j++)
    {
        for(int i=0; i<p.nx; i++)
        {
            tx = (-(p.nx-1)/2.0 + i) * p.dx;
            ty = (-(p.ny-1)/2.0 + j) * p.dx;

            flag = (tx>(-cyp.rad-1.1*p.dx) && tx<(-cyp.rad+0.1*p.dx)) | (tx>(cyp.rad-0.1*p.dx) && tx<(cyp.rad+1.1*p.dx));
            if (flag)
            {
                if (ty>=-1.0e-10)
                    ofs<<NAN<<" "<<NAN<<" "<<0.0<<" "<<NAN<<endl;
                else
                {
                    ofs<<ba.disp_x[num]<<" "<<ba.disp_y[num]<<" "<<ba.disp_z[num]<<" "<<ba.dmg[num]<<endl;
                    num++;
                }
            } else {
                ofs<<ba.disp_x[num]<<" "<<ba.disp_y[num]<<" "<<ba.disp_z[num]<<" "<<ba.dmg[num]<<endl;
                num++;
            }
        }
    }
}

