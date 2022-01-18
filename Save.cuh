//
// Created by wxm on 2021/12/3.
//

#ifndef PD_PARALLEL_SAVE_CUH
#define PD_PARALLEL_SAVE_CUH
#include "Base.cuh"
void save_disp_gpu(Dselect &d, BAtom &ba, const string FILE="disp.txt");
void save_disp_cpu(Dselect &d, BAtom &ba, const string FILE="disp.txt");
void save_kw_gpu( BAtom &ba, const string FILE="disp.txt");
void save_kw_cpu( BAtom &ba, const string FILE="disp.txt");
#endif //PD_PARALLEL_SAVE_CUH
