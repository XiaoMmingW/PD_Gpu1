
#include "Solve_Plate_Tensile.cuh"
#include "Solve_Plate_Crack_Grow.cuh"
#include "Solve_Kalthoff_Winkler.cuh"

__constant__ device_parameter dp;
__constant__ Cylinder_Parameter dcyp;
device_parameter p;
Cylinder_Parameter cyp;

int main(int argc, char ** argv) {

    Dselect d{2,3,1,3};
    int mode = 3;
    real dx = 0.0000;
    cudaSetDevice(1);
    if (argc>1)
    {
        d.device = atoi(argv[1]);
        if (argc>2)
            mode = atoi(argv[2]);
        if (argc>3)
            dx = atoi(argv[3]);
    }



    switch (mode)
    {
        case 1 :
            solve_plate_tensile(d, dx);
            break;
        case 2 :
            solve_plate_crack_grow(d, dx);
            break;
        case 3:
            solve_kalthoff_winkler(d, dx);
            break;
        default:
            break;
    }







    return 0;
}
