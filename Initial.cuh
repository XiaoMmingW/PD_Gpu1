//
// Created by wxm on 2021/12/3.
//

#ifndef PD_PARALLEL_INITIAL_CUH
#define PD_PARALLEL_INITIAL_CUH
#include "Base.cuh"
void base_integrate_initial_gpu(Dselect &d,BAtom &ba, BBond &bb);
void static_initial_bond_gpu(Dselect &d,BAtom &ba, BBond &bb);
void base_integrate_initial_cpu(Dselect &d,BAtom &ba, BBond &bb);
void static_initial_bond_cpu(Dselect &d,BAtom &ba, BBond &bb);
#endif //PD_PARALLEL_INITIAL_CUH
