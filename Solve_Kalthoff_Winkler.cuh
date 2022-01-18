//
// Created by wxm on 2021/12/7.
//

#ifndef PD_PARALLEL_SOLVE_KALTHOFF_WINKLER_CUH
#define PD_PARALLEL_SOLVE_KALTHOFF_WINKLER_CUH
#include "Base.cuh"
#include "Memory.cuh"
#include "Coord.cuh"
#include "Neighbor.cuh"
#include "Surf_Corr.cuh"
#include "Initial.cuh"
#include "Force_State.cuh"
#include "Force_Bond.cuh"
#include "Integrate.cuh"
#include "Base_Function.cuh"
#include "Save.cuh"
void solve_kalthoff_winkler(Dselect &d, real dx);
#endif //PD_PARALLEL_SOLVE_KALTHOFF_WINKLER_CUH
