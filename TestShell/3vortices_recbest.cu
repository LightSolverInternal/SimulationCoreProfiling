#include <stdio.h>
#include <stdlib.h>
#include "curand.h"
#include "SimulationCore.h"
#include <time.h>

int propogate_states_3vortices_recbest(
    int nlasers,                        // Number of lasers
    int nstates,                        // Number of states to run in parallel
    float* jijmat,                      // [nlasers/3][nlasers/3] problem interaction matrix in Ising format, zero on diagonal
    float* himat,                       // [nlasers/3] problem external fields in Ising format
    cuFloatComplex* coupmat,            // [nlasers][nlasers] Coupling matrix
    cuFloatComplex* states,             // [nlasers][nstates] IN: Starting states matrix, OUT: Ending states matrix
    float* gain,                        // [nlasers][nstates] IN: Starting gains matrix, OUT: Ending gains matrix
    cuFloatComplex* recstates,          // [nlasers/3+1][nstates] Recorded states
    float* recstates_binary,            // [nlasers/3+1][nstates] Recorded states in +1/-1
    float* recgain,                     // [nlasers/3+1][nstates] Recorded gains
    float* recenergy,                   // [nstates] Recorded energy
    int* recindex,                      // [nstates] Recorded index
    int* reccounter,                    // [nstates] Recorded index
    float energyoffset,                 // Energy offset for probmat
    float energystop,
    float pump_max,
    float pump_tau,
    float pump_tresh,
    float ampsat,                       // Gain saturation amplitude
    int niter,                          // Number of steps    
    int rounds_per_step,                // Number of round trips per calculation time step
    float noiseamp                      // Noise amplitude. 0 for no noise.
)
{
    // Initializations
    cudaError_t cudaStatus;
    curandStatus_t randStatus;
    cublasStatus_t cublasstatus;
    cublasHandle_t handle;
    int ret = 0;
    cuFloatComplex* gpu_coupmat1;
    cuFloatComplex* gpu_coupmat2;
    cuFloatComplex* gpu_statesr;
    cuFloatComplex* gpu_statesr_conj;
    cuFloatComplex* gpu_statesr_temp;
    float* gpu_jijmat;
    float* gpu_himat;
    float* gpu_states_binary;
    float* gpu_states_temp;
    float* gpu_states_energy;
    float* gpu_gainsr;
    cuFloatComplex* coupmat1;
    cuFloatComplex* coupmat2;
    cuFloatComplex* statesr;
    cuFloatComplex* statesr_conj;
    float* jijmat_ext;
    float* himat_ext;
    float* states_energy;
    float* gainsr;
    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(0.0f, 0.0f);
    const float falpha = 1.0f;
    const float fbeta = 0.0f;
    const int szBlockMNW = 32;
    int szGridN;
    int szGridM;
    int nc;
    float* gpu_noise_amp;
    float* gpu_noise_phase;
    curandGenerator_t randgenerator;
    int state_array_size;
    int coupmat_size;
    char logLine[256];

    if (noiseamp > 0) {
        // Create random generator and seed.
        randStatus = curandCreateGenerator(&randgenerator, CURAND_RNG_PSEUDO_DEFAULT);
        if (randStatus != 0) {
            ret = -1;
            printf("curandCreateGenerator error: %d\n", randStatus);
            return ret;
        }
        randStatus = curandSetPseudoRandomGeneratorSeed(randgenerator, (int)(time(NULL))+ (int)(clock())); // Randomize the seed according to the time
        if (randStatus != 0) {
            ret = -1;
            printf("curandSetPseudoRandomGeneratorSeed error: %d\n", randStatus);
            return ret;
        }
        noiseamp = noiseamp / ampsat; // We define the noise amplitude relative to the saturation amplitude. There is no reason not to set ampsat to 1 (the defualt), but just to make sure in case someone does something weird.
    }

    // Reduce number rows
    nc = nlasers / 3 + 1;
    if (((nc - 1) * 3) != nlasers) {
        ret = -1;
        printf("Number of lasers is not 3 X number of spins: nlasers = %d at %d\n", nlasers, ret);
        return ret;
    }

    // Determined optimal sizes of blocks
    szGridN = (nstates + szBlockMNW - 1) / szBlockMNW;
    szGridM = (nc + szBlockMNW - 1) / szBlockMNW;
    printf("Simulation core kernel configuration szGridM(szBlockM), szGridN(szBlockN): %d(%d), %d(%d)\n"
        , szGridM, szBlockMNW, szGridN, szBlockMNW);

    // Define kernel grid, block and shared memory parameters   
    dim3 dimBlock(szBlockMNW, szBlockMNW, 1);
    dim3 dimGrid1(szGridM, szGridN, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        ret = -10;
        printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
        return ret;
    }

    // Create a handle for CUBLAS
    cublasstatus = cublasCreate(&handle);
    if (cublasstatus != CUBLAS_STATUS_SUCCESS) {
        ret = -11;
        printf("cuBLAS error %d at %d\n", (int)cublasstatus, ret);
        return ret;
    }
    state_array_size = nc * nstates;
    coupmat_size = nc * nc;

    // Allocate GPU memory
    cudaMalloc((void**)&gpu_coupmat1, sizeof(cuFloatComplex) * coupmat_size);
    cudaMalloc((void**)&gpu_coupmat2, sizeof(cuFloatComplex) * coupmat_size);
    cudaMalloc((void**)&gpu_statesr, sizeof(cuFloatComplex) * state_array_size);
    cudaMalloc((void**)&gpu_statesr_conj, sizeof(cuFloatComplex) * state_array_size);
    cudaMalloc((void**)&gpu_statesr_temp, sizeof(cuFloatComplex) * state_array_size);
    cudaMalloc((void**)&gpu_jijmat, sizeof(float) * coupmat_size);
    cudaMalloc((void**)&gpu_himat, sizeof(float) * nc);
    cudaMalloc((void**)&gpu_states_binary, sizeof(float) * state_array_size);
    cudaMalloc((void**)&gpu_states_temp, sizeof(float) * state_array_size);
    cudaMalloc((void**)&gpu_states_energy, sizeof(float) * nstates);
    cudaMalloc((void**)&gpu_gainsr, sizeof(float) * state_array_size);
    if (noiseamp > 0.0f) {
        cudaMalloc((void**)&gpu_noise_amp, sizeof(float) * (state_array_size + (0x1 & state_array_size)));
        cudaMalloc((void**)&gpu_noise_phase, sizeof(float) * state_array_size);
    }
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        ret = -20;
        printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
        goto exit;
    }

    // Allocate CPU memory
    coupmat1 = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * coupmat_size);
    coupmat2 = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * coupmat_size);
    statesr = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * state_array_size);
    statesr_conj = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * state_array_size);
    jijmat_ext = (float*)malloc(sizeof(float) * coupmat_size);
    himat_ext = (float*)malloc(sizeof(float) * nc);
    states_energy = (float*)malloc(sizeof(float) * nstates);
    gainsr = (float*)malloc(sizeof(float) * state_array_size);
    if ((coupmat1 == NULL) || (coupmat2 == NULL) || (statesr == NULL) || (states_energy == NULL) || (gainsr == NULL) ) {
        ret = -21;
        printf("Memory allocation error at %d\n", ret);
        goto exit;
    }

    // Populate CPU arrays
    coupmat1->x = coupmat->x * (float)(nc - 1);
    coupmat1->y = coupmat->y * (float)(nc - 1);
    coupmat2->x = 0.0f;
    coupmat2->y = 0.0f;
    *himat_ext = 0.0f;
    *jijmat_ext = 0.0f;
    for (size_t i1 = 1; i1 < nc; i1++) {
        // 1st column (ref to all)
        (coupmat1 + i1)->x = (coupmat + i1 * 3 - 2)->x * (float)(nc - 1);
        (coupmat1 + i1)->y = (coupmat + i1 * 3 - 2)->y * (float)(nc - 1);
        (coupmat2 + i1)->x = 0.0f;
        (coupmat2 + i1)->y = 0.0f;
        *(jijmat_ext + i1) = 0.0f;
        *(himat_ext + i1) = *(himat + i1 - 1);
        // 1st row (all to ref)
        (coupmat1 + i1 * nc)->x = (coupmat + (i1 * 3 - 2) * nlasers)->x;
        (coupmat1 + i1 * nc)->y = (coupmat + (i1 * 3 - 2) * nlasers)->y;
        (coupmat2 + i1 * nc)->x = (coupmat + (i1 * 3 - 1) * nlasers)->x;
        (coupmat2 + i1 * nc)->y = (coupmat + (i1 * 3 - 1) * nlasers)->y;
        *(jijmat_ext + i1 * nc) = 0.0f;
    }
    for (size_t i2 = 1; i2 < nc; i2++) { // column
        for (size_t i1 = 1; i1 < nc; i1++) { //row
            (coupmat1 + i2 * nc + i1)->x = (coupmat + (i2 * 3 - 2) * nlasers + i1 * 3 - 2)->x;
            (coupmat1 + i2 * nc + i1)->y = (coupmat + (i2 * 3 - 2) * nlasers + i1 * 3 - 2)->y;
            (coupmat2 + i2 * nc + i1)->x = (coupmat + (i2 * 3 - 1) * nlasers + i1 * 3 - 2)->x;
            (coupmat2 + i2 * nc + i1)->y = (coupmat + (i2 * 3 - 1) * nlasers + i1 * 3 - 2)->y;
            *(jijmat_ext + i2 * nc + i1) = *(jijmat + (i2 - 1) * (nc - 1) + i1 - 1);
        }
    }

    for (size_t i2 = 0; i2 < nstates; i2++) {
        (statesr_conj + i2 * nc)->x = 0.0f; 
        (statesr + i2 * nc)->x = (states + i2 * nlasers)->x;
        (statesr_conj + i2 * nc)->y = 0.0f; 
        (statesr + i2 * nc)->y = (states + i2 * nlasers)->y;
		*(gainsr + i2 * nc) = *(gain + i2 * nlasers);
		for (size_t i1 = 1; i1 < nc; i1++) {
            (statesr_conj + i2 * nc + i1)->x = (statesr + i2 * nc + i1)->x = (states + i2 * nlasers + i1 * 3 - 2)->x;
            (statesr_conj + i2 * nc + i1)->y = -((statesr + i2 * nc + i1)->y = (states + i2 * nlasers + i1 * 3 - 2)->y);
			*(gainsr + i2 * nc + i1) = *(gain + i2 * nlasers + i1 * 3 - 2);
		}
	}

    for (int k1 = 0; k1 < nstates; k1++) *(recenergy + k1) = 1e34f;


     // Copy CPU arrays to GPU
    cudaMemcpy((void*)gpu_coupmat1, (const void*)coupmat1, sizeof(cuFloatComplex) * coupmat_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_coupmat2, (const void*)coupmat2, sizeof(cuFloatComplex) * coupmat_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_statesr, (const void*)statesr, sizeof(cuFloatComplex) * state_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_statesr_conj, (const void*)statesr_conj, sizeof(cuFloatComplex) * state_array_size, cudaMemcpyHostToDevice); 
    cudaMemcpy((void*)gpu_jijmat, (const void*)jijmat_ext, sizeof(float) * coupmat_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_himat, (const void*)himat_ext, sizeof(float) * nc, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_gainsr, (const void*)gainsr, sizeof(float) * state_array_size, cudaMemcpyHostToDevice);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        ret = -30;
        printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
        goto exit;
    }

    // Propogation steps loop
    for (int i1 = 0; i1 < niter; i1++) {

        cublasstatus = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nc, nstates, nc, &alpha, gpu_coupmat1, nc, gpu_statesr, nc, &beta, gpu_statesr_temp, nc);
        if (cublasstatus != CUBLAS_STATUS_SUCCESS) {
            ret = -41;
            printf("cuBLAS error %d at %d\n", (int)cublasstatus, ret);
            return ret;
        }

        cublasstatus = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nc, nstates, nc, &alpha, gpu_coupmat2, nc, gpu_statesr_conj, nc, &alpha, gpu_statesr_temp, nc);
        if (cublasstatus != CUBLAS_STATUS_SUCCESS) {
            ret = -42;
            printf("cuBLAS error %d at %d\n", (int)cublasstatus, ret);
            return ret;
        }


        //Compute pump state
        float p = pump_tresh + ((pump_max - pump_tresh) * (1 - expf(-((float)(i1 * rounds_per_step)) / pump_tau)));
        //Invoke kernel to aggragate sub-blocks and advance gains
        // Could use function pointer instead of an if here.
        if (noiseamp == 0.0f) {
            // No noise.
            stepcublas3 << < dimBlock, dimGrid1 >> > (nc, nstates, gpu_statesr_temp, gpu_statesr, gpu_statesr_conj, gpu_gainsr, gpu_states_binary, p, ampsat, rounds_per_step);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                ret = -43;
                printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
                goto exit;
            }
        }
        else {
            // With noise
            // Generate random numbers
            
            randStatus = curandGenerateUniform(randgenerator, gpu_noise_phase, state_array_size);
            if (randStatus != 0) {
                ret = -1;
                printf("curandGenerateUniform error: %d\n", randStatus);
                return ret;
            }

            randStatus = curandGenerateNormal(randgenerator, gpu_noise_amp, state_array_size + (0x1 & state_array_size), 0.0, 1.0);
            if (randStatus != 0) {
                ret = -1;
                printf("curandGenerateNormal error: %d\n", randStatus);
                return ret;
            }

            stepcublas3noise << < dimBlock, dimGrid1 >> > (nc, nstates, gpu_statesr_temp, gpu_statesr, gpu_statesr_conj, gpu_gainsr, gpu_states_binary, p, ampsat, rounds_per_step, noiseamp, gpu_noise_phase, gpu_noise_amp);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                ret = -51;
                printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
                goto exit;
            }
        }


         // Compute sum_j(J_ij * S_j)
        cublasstatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nc, nstates, nc, &falpha, gpu_jijmat, nc, gpu_states_binary, nc, &fbeta, gpu_states_temp, nc);
        if (cublasstatus != CUBLAS_STATUS_SUCCESS) {
            ret = -44;
            printf("cuBLAS error %d at %d\n", (int)cublasstatus, ret);
            return ret;
        }


        // Compute sum_i(S_i * (h_i + sum_j(J_ij * S_j)) 
        stepcublas4 << < szBlockMNW, szGridN >> > (nc, nstates, gpu_himat, gpu_states_binary, gpu_states_temp, gpu_states_energy);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            ret = -45;
            printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
            goto exit;
        }
        cudaMemcpy((void*)states_energy, (const void*)gpu_states_energy, sizeof(float)* nstates, cudaMemcpyDeviceToHost);


        // Check energy
        int exitflag = false;
        for (int k1 = 0; k1 < nstates; k1++) {
            float temp = ((*(states_energy + k1) += energyoffset) - *(recenergy + k1)) / (fabsf(*(recenergy + k1)) + 1e-6f);
            if (temp < -1e-6f) {
                *(recenergy + k1) = *(states_energy + k1);
                *(recindex + k1) = i1;
                *(reccounter + k1) = 1;
                cudaMemcpy((void*)(recstates + k1 * nc), (const void*)(gpu_statesr + k1 * nc), sizeof(cuFloatComplex) * nc, cudaMemcpyDeviceToHost);
                cudaMemcpy((void*)(recstates_binary + k1 * nc), (const void*)(gpu_states_binary + k1 * nc), sizeof(float) * nc, cudaMemcpyDeviceToHost);
                cudaMemcpy((void*)(recgain + k1 * nc), (const void*)(gpu_gainsr + k1 * nc), sizeof(float) * nc, cudaMemcpyDeviceToHost);
                if (*(recenergy + k1) <= energystop) { exitflag = true; }
            }
            else if (temp < 1e-6f) *(reccounter + k1) += 1;
        }
        if (exitflag) break;

    }

    // Copy data from GPU to CPU
    cudaMemcpy((void*)statesr, (const void*)gpu_statesr, sizeof(cuFloatComplex) * state_array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)gainsr, (const void*)gpu_gainsr, sizeof(float) * state_array_size, cudaMemcpyDeviceToHost);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        ret = -50;
        printf("CUDA error %d at %d\n", (int)cudaStatus, ret);
        goto exit;
    }

    for (size_t i2 = 0; i2 < nstates; i2++) {
        for (size_t i1 = 1; i1 < nc; i1++) {
            (states + i2 * nlasers + i1 * 3 - 3)->x = (statesr + i2 * nc)->x;
            (states + i2 * nlasers + i1 * 3 - 3)->y = (statesr + i2 * nc)->y;
            *(gain + i2 * nlasers + i1 * 3 - 3) = *(gainsr + i2 * nc);
            (states + i2 * nlasers + i1 * 3 - 1)->x = (states + i2 * nlasers + i1 * 3 - 2)->x = (statesr + i2 * nc + i1)->x;
            (states + i2 * nlasers + i1 * 3 - 1)->y = -((states + i2 * nlasers + i1 * 3 - 2)->y = (statesr + i2 * nc + i1)->y);
            *(gain + i2 * nlasers + i1 * 3 - 2) = *(gain + i2 * nlasers + i1 * 3 - 1) = *(gainsr + i2 * nc + i1);
        }
    }

exit:
    // Destroy the handle
    cublasDestroy(handle);
    // Free GPU memory
    cudaFree((void*)gpu_coupmat1);
    cudaFree((void*)gpu_coupmat2);
    cudaFree((void*)gpu_statesr);
    cudaFree((void*)gpu_statesr_conj);
    cudaFree((void*)gpu_statesr_temp);
    cudaFree((void*)gpu_jijmat);
    cudaFree((void*)gpu_himat);
    cudaFree((void*)gpu_states_binary);
    cudaFree((void*)gpu_states_temp);
    cudaFree((void*)gpu_states_energy);
    cudaFree((void*)gpu_gainsr);
    if (noiseamp > 0.0f) {
        curandDestroyGenerator(randgenerator);
        cudaFree((void*)gpu_noise_amp);
        cudaFree((void*)gpu_noise_phase);
    }
    // Free  CPU memory
    free((void*)coupmat1);
    free((void*)coupmat2);
    free((void*)statesr);
    free((void*)statesr_conj);
    free((void*)states_energy);
    free((void*)gainsr);

    return ret;
}
