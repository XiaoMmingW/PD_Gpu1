//
// Created by wxm on 2021/12/6.
//

#ifndef PD_PARALLEL_SOLVE_PLATE_CRACK_GROW_CUH
#define PD_PARALLEL_SOLVE_PLATE_CRACK_GROW_CUH
#include "Base.cuh"
#include "Memory.cuh"
#include "Coord.cuh"
#include "Neighbor.cuh"
#include "Surf_Corr.cuh"
#include "Initial.cuh"
#include "Force_State.cuh"
#include "Integrate.cuh"
#include "Base_Function.cuh"
#include "Save.cuh"

void solve_plate_crack_grow(Dselect &d, real dx=0.0);
#endif //PD_PARALLEL_SOLVE_PLATE_CRACK_GROW_CUH
