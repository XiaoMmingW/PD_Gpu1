//
// Created by wxm on 2021/12/5.
//

#ifndef PD_PARALLEL_SOLVE_PLATE_TENSILE_CUH
#define PD_PARALLEL_SOLVE_PLATE_TENSILE_CUH
#include "Base.cuh"
#include "Memory.cuh"
#include "Coord.cuh"
#include "Neighbor.cuh"
#include "Surf_Corr.cuh"
#include "Initial.cuh"
#include "Force_Bond.cuh"
#include "Integrate.cuh"
#include "Base_Function.cuh"
#include "Save.cuh"
void solve_plate_tensile(Dselect &d, real dx=0.0);

#endif //PD_PARALLEL_SOLVE_PLATE_TENSILE_CUH
