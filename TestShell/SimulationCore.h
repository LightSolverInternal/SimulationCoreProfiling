#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>


void write_log(char* text, bool newLine = true);


__global__ void stepcublas1(int, int, cuFloatComplex*, cuFloatComplex*, float*, float, float, int);
__global__ void stepcublas1ns(int, int, cuFloatComplex*, cuFloatComplex*, float*, float, float, int);
__global__ void stepcublas2(int, int, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, float, float, int);
__global__ void stepcublas3(int, int, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, float*, float, float,int);
__global__ void stepcublas3noise(int, int, cuFloatComplex*, cuFloatComplex*, cuFloatComplex*, float*, float*, float, float, int, float, float*, float*);
__global__ void stepcublas4(int, int, float*, float*, float*, float*);
__global__ void stepcublas5(int, int, float*, float*, float*, float*, float*, float*, float*);
__global__ void stepcublas6(int, int, float*, float*, float*, float*);
extern "C" int propogate_states_3vortices_recbest(int, int, float*, float*, cuFloatComplex*, cuFloatComplex*, float*, cuFloatComplex*,
            float*, float*, float*, int*, int*, float, float, float, float, float, float, int, int, float);
