
#include "SimulationCore.h"


int main()
{

    cudaError_t cudaStatus;

    int nlasers = 900;
    int nstates = 1;
    float* jijmat;
    float* himat;
    cuFloatComplex* coupmat;
    cuFloatComplex* states;
    float* gain;
    cuFloatComplex* recstates;
    float* recstates_binary;
    float* recgain;
    float* recenergy;
    int* recindex;
    int* reccounter;
    float energyoffset = 0.0f;
    float energystop = -1e6f;
    float pump_max = 0.3424f;
    float pump_tau = 100.0f;
    float pump_tresh = 0.2553f;
    float ampsat = 1.0f;
    int niter = 500;
    int rounds_per_step = 1;
    float noiseamp = 0.0f;
    
    // Allocate CPU memory
    jijmat = (float*)malloc(sizeof(float) * (nlasers * nlasers / 9)); 
    himat = (float*)malloc(sizeof(float) * (nlasers / 3));
    coupmat = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * (nlasers * nlasers));
    states = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * (nlasers * nstates));
    gain = (float*)malloc(sizeof(float) * (nlasers * nstates));

    recstates = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * (nlasers/3+1) * nstates);
    recstates_binary = (float*)malloc(sizeof(float) * (nlasers / 3 + 1) * nstates);
    recgain = (float*)malloc(sizeof(float) * (nlasers / 3 + 1) * nstates);
    recenergy = (float*)malloc(sizeof(float) * nstates);
    recindex = (int*)malloc(sizeof(int) * nstates);
    reccounter = (int*)malloc(sizeof(int) * nstates);

    if ((jijmat == NULL) || (himat == NULL) || (coupmat == NULL) || (states == NULL) || (gain == NULL) || 
        (recstates == NULL) || (recstates_binary == NULL) || (recgain == NULL) || (recenergy == NULL) || (recindex == NULL) || (reccounter == NULL)) {
        printf("Memory allocation error at %d\n", -5);
        return -5;
    }

    for (size_t i1 = 0; i1 < nlasers; i1++) {
        for (size_t i2 = 0; i2 < nlasers; i2++) {
            (coupmat + i2 * nlasers + i1)->x = 1.0f;
            (coupmat + i2 * nlasers + i1)->y = 0.1f;
        }
        for (size_t i2 = 0; i2 < nstates; i2++) {
            (states + i2 * nlasers + i1)->x = 1.0f;
            (states + i2 * nlasers + i1)->y = 0.1f;
            *(gain + i2 * nlasers + i1) = 1.0f;
        }
    }

    for (size_t i1 = 0; i1 < nlasers/3; i1++) {
        for (size_t i2 = 0; i2 < nlasers/3; i2++) {
            *(jijmat + i2 * nlasers/3 + i1) = 1.0f;
        }
        *(himat + i1) = 1.0f;
    }


    int out = propogate_states_3vortices_recbest(
        nlasers,                            // int Number of lasers
        nstates,                            // int Number of states to run in parallel
        jijmat,                             // float* [nlasers/3][nlasers/3] problem interaction matrix in Ising format, zero on diagonal
        himat,                              // float* [nlasers/3] problem external fields in Ising format
        coupmat,                            // cuFloatComplex * [nlasers][nlasers] Coupling matrix
        states,                             // cuFloatComplex * [nlasers][nstates] IN: Starting states matrix, OUT: Ending states matrix
        gain,                               // float* [nlasers][nstates] IN: Starting gains matrix, OUT: Ending gains matrix
        recstates,                          // cuFloatComplex * [nlasers/3+1][nstates] Recorded states
        recstates_binary,                   // float* [nlasers/3+1][nstates] Recorded states in +1/-1
        recgain,                            // float* [nlasers/3+1][nstates] Recorded gains
        recenergy,                          // float* [nstates] Recorded energy
        recindex,                           // int* [nstates] Recorded index
        reccounter,                         // int* [nstates] Recorded index
        energyoffset,                       // float Energy offset for probmat
        energystop,                         // float 
        pump_max,                           // float 
        pump_tau,                           // float 
        pump_tresh,                         // float 
        ampsat,                             // float Gain saturation amplitude
        niter,                              // int Number of steps    
        rounds_per_step,                    // int Number of round trips per calculation time step
        noiseamp                            // float Noise amplitude. 0 for no noise.
    );

    printf("Exit \n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
 
    free(jijmat);
    free(himat);
    free(coupmat);
    free(states);
    free(gain);

    free(recstates);
    free(recstates_binary);
    free(recgain);
    free(recenergy);
    free(recindex);
    free(reccounter);
    
    return 0;
}
