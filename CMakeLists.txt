cmake_minimum_required(VERSION 3.20)
project(Pd_parallel CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Pd_parallel main.cu Base.cuh Error.cuh Coord.cu Coord.cuh Memory.cu Memory.cuh Neighbor.cu Neighbor.cuh Base_Function.cu
        Base_Function.cuh Surf_Corr.cu Surf_Corr.cuh Integrate.cu Integrate.cuh Reduce.cu Reduce.cuh Initial.cu Initial.cuh Save.cu
        Save.cuh Force.cu Force.cuh Force_Bond.cu Force_Bond.cuh Force_State.cu Force_State.cuh Solve_Plate_Tensile.cu Solve_Plate_Tensile.cuh Solve_Plate_Crack_Grow.cu Solve_Plate_Crack_Grow.cuh Solve_Kalthoff_Winkler.cu Solve_Kalthoff_Winkler.cuh)

set_target_properties(Pd_parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
