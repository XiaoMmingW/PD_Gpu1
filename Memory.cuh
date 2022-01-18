//
// Created by wxm on 2021/12/2.
//

#ifndef PD_PARALLEL_MEMORY_CUH
#define PD_PARALLEL_MEMORY_CUH
#include "Base.cuh"
void Base_Allocate( BAtom &ba, BBond &bb, const Dselect &d);
void Base_Free(BAtom &ba, BBond &bb, const Dselect &d);
void cy_allocate(const Dselect &d, Cylinder &cy);
void cy_free(const Dselect &d, Cylinder &cy);
#endif //PD_PARALLEL_MEMORY_CUH
