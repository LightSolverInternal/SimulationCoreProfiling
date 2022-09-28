#include "SimulationCore.h"
#include "curand_kernel.h"
#include "math_constants.h"


__global__ void stepcublas1(
	int nlasers,
	int nstates,
	cuFloatComplex* gpu_states_temp,
	cuFloatComplex* gpu_states,
	float* gpu_gains,
	float p,
	float ampsat,
	int rounds_per_step)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;	// m indedx in gpu_coupmat[m,w]
	int n = blockIdx.y * blockDim.y + threadIdx.y;	// n indedx in gpu_states[w,n]

	if ((m < nlasers) && (n < nstates)) {
		cuFloatComplex* statep_t = gpu_states_temp + n * nlasers + m;
		cuFloatComplex* statep = gpu_states + n * nlasers + m;
		float* gainp = gpu_gains + n * nlasers + m;
		float gainv = expf(*gainp);
		int modl = m - (m / 3) * 3;
		if (modl == 0) {
			statep->x = fabsf( statep_t->x * gainv);
			statep->y = 0.0f;
		}
		else {
			statep->x = ((statep_t - modl + 1)->x + (statep_t - modl + 2)->x) * 0.5f * gainv;
			statep->y = (3 - (float)modl * 2) * ((statep_t - modl + 1)->y - (statep_t - modl + 2)->y) * 0.5f * gainv;
		}
		*gainp += rounds_per_step * (6.0f / 1e4f) * (p - *gainp * ((sqrtf(statep->x * statep->x + statep->y * statep->y) / ampsat) + 1.0f));
	}
}

__global__ void stepcublas1ns(
	int nlasers,
	int nstates,
	cuFloatComplex* gpu_states_temp,
	cuFloatComplex* gpu_states,
	float* gpu_gains,
	float p,
	float ampsat,
	int rounds_per_step)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;	// m indedx in gpu_coupmat[m,w]
	int n = blockIdx.y * blockDim.y + threadIdx.y;	// n indedx in gpu_states[w,n]

	if ((m < nlasers) && (n < nstates)) {
		cuFloatComplex* statep_t = gpu_states_temp + n * nlasers + m;
		cuFloatComplex* statep = gpu_states + n * nlasers + m;
		float* gainp = gpu_gains + n * nlasers + m;
		float gainv = expf(*gainp);
		statep->x = statep_t->x * gainv;
		statep->y = statep_t->y * gainv;
		*gainp += rounds_per_step * (6.0f / 1e4f) * (p - *gainp * ((sqrtf(statep->x * statep->x + statep->y * statep->y) / ampsat) + 1.0f));
	}
}



__global__ void stepcublas2(
	int nc,
	int nstates,
	cuFloatComplex* gpu_states_temp,
	cuFloatComplex* gpu_states,
	cuFloatComplex* gpu_states_conj,
	float* gpu_gains,
	float p,
	float ampsat,
	int rounds_per_step)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;	// m indedx in gpu_coupmat[m,w]
	int n = blockIdx.y * blockDim.y + threadIdx.y;	// n indedx in gpu_states[w,n]

	if ((m < nc) && (n < nstates)) {
		cuFloatComplex* statep_t = gpu_states_temp + n * nc + m;
		cuFloatComplex* statep = gpu_states + n * nc + m;
		cuFloatComplex* statep_c = gpu_states_conj + n * nc + m;
		float* gainp = gpu_gains + n * nc + m;
		float gainv = expf(*gainp);
		statep_c->x = (statep->x = statep_t->x * gainv);
		statep_c->y = -(statep->y = statep_t->y * gainv * (float)(m > 0));
		if (m == 0) statep->x = fabsf(statep->x);
		*gainp += rounds_per_step * (6.0f / 1e4f) * (p - *gainp * ((sqrtf(statep->x * statep->x + statep->y * statep->y) / ampsat) + 1.0f));
	}
}


__global__ void stepcublas3(
	int nc,
	int nstates,
	cuFloatComplex* gpu_states_temp,
	cuFloatComplex* gpu_states,
	cuFloatComplex* gpu_statesr_conj,
	float* gpu_gains,
	float* gpu_states_binary,
	float p,
	float ampsat,
	int rounds_per_step)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;	// m indedx in gpu_coupmat[m,w]
	int n = blockIdx.y * blockDim.y + threadIdx.y;	// n indedx in gpu_states[w,n]

	if ((m < nc) && (n < nstates)) {
		cuFloatComplex* statep_t = gpu_states_temp + n * nc + m;
		cuFloatComplex* statep = gpu_states + n * nc + m;
		cuFloatComplex* statep_c = gpu_statesr_conj + n * nc + m;
		float* gainp = gpu_gains + n * nc + m;
		float gainv = expf(*gainp);
		statep_c->x = (statep->x = statep_t->x * gainv);
		statep_c->y = -(statep->y = statep_t->y * gainv * (float)(m > 0));
		if (m == 0) statep_c->x = (statep->x = fabsf(statep->x));
		*gainp += rounds_per_step * (6.0f / 1e4f) * (p - *gainp * ((sqrtf(statep->x * statep->x + statep->y * statep->y) / ampsat) + 1.0f));
		*(gpu_states_binary + n * nc + m) = 2.0f * (float)(statep->y >= 0.0f) - 1.0f;
	}
}

__global__ void stepcublas3noise(
	int nc,
	int nstates,
	cuFloatComplex* gpu_states_temp,
	cuFloatComplex* gpu_states,
	cuFloatComplex* gpu_statesr_conj,
	float* gpu_gains,
	float* gpu_states_binary,
	float p,
	float ampsat,
	int rounds_per_step,
	float noiseamp,
	float* gpu_noise_phase,
	float* gpu_noise_amp)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;	// m indedx in gpu_coupmat[m,w]
	int n = blockIdx.y * blockDim.y + threadIdx.y;	// n indedx in gpu_states[w,n]
	if ((m < nc) && (n < nstates)) {
		cuFloatComplex* statep_t = gpu_states_temp + n * nc + m;
		cuFloatComplex* statep = gpu_states + n * nc + m;
		cuFloatComplex* statep_c = gpu_statesr_conj + n * nc + m;
		float* gainp = gpu_gains + n * nc + m;
		float gainv = expf(*gainp);
		float noiserandamp = *(gpu_noise_amp + n * nc + m);
		float noiserandphase = *(gpu_noise_phase + n * nc + m);
		// The noise is ABSOLUTE and NOT RELATIVE.
		statep->x = statep_t->x * gainv + noiseamp * noiserandamp * cosf(2 * CUDART_PI_F * noiserandphase);
		statep_c->x = statep->x;
		statep->y = (statep_t->y * gainv + noiseamp * noiserandamp * sinf(2 * CUDART_PI_F * noiserandphase)) * (float)(m > 0);
		statep_c->y = -statep->y;
		if (m == 0) statep_c->x = (statep->x = fabsf(statep->x));
		*gainp += rounds_per_step * (6.0f / 1e4f) * (p - *gainp * ((sqrtf(statep->x * statep->x + statep->y * statep->y) / ampsat) + 1.0f));
		*(gpu_states_binary + n * nc + m) = 2.0f * (float)(statep->y >= 0.0f) - 1.0f;
	}
}

// Compute sum_i(S_i * (h_i + sum_j(J_ij * S_j))
__global__ void stepcublas4(
	int nc,
	int nstates,
	float* gpu_himat,
	float* gpu_states_binary,
	float* gpu_states_temp,
	float* gpu_states_energy)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;	// n indedx in gpu_states[w,n]

	if (n < nstates) {
		float* energy = gpu_states_energy + n;
		float* binary = gpu_states_binary + n * nc + 1;
		float* temp = gpu_states_temp + n * nc + 1;
		float* him = gpu_himat + 1;
		*energy = 0.0f;
		for (int k1 = 1; k1 < nc; k1++) {
			*energy += *(binary++) * (*(him++) + *(temp++));
		}
	}
}

// states = (aux + (states * a + b) * states + c) * gain
__global__ void stepcublas5(
	int nspins,
	int nstates,
	float* gpu_states,
	float* gpu_states_binary,
	float* gpu_aux,
	float* gpu_acoeff,
	float* gpu_bcoeff,
	float* gpu_ccoeff,
	float* gpu_gain)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;	// n indedx of spin

	if (n < nspins) {
		float a = *(gpu_acoeff + n);
		float b = *(gpu_bcoeff + n);
		float c = *(gpu_ccoeff + n);
		float gain = *(gpu_gain + n);
		for (int k1 = 0; k1 < nstates; k1++) {
			int offset = k1 * nspins + n;
			float st = *(gpu_states + offset);
			*(gpu_states_binary + offset) = -1.0f + 2.0f *
				((*(gpu_states + offset) += (*(gpu_aux + offset) + c + st * (b + st * a)) * gain) >= 0.0f);
		}
	}
}

// gpu_aux = (gpu_aux + gpu_hi) * gpu_states_binary
__global__ void stepcublas6(
	int nspins,
	int nstates,
	float* gpu_aux,
	float* gpu_states_binary,
	float* gpu_hicoeff,
	float* gpu_energy)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;	// n indedx of spin

	if (n < nstates) {
		*(gpu_energy + n) = 0.0f;
		int offset = nspins * n;
		for (int k1 = 0; k1 < nspins; k1++) {
			*(gpu_energy + n) += (*(gpu_aux + offset + k1) + *(gpu_hicoeff + k1)) * *(gpu_states_binary + offset + k1);
		}
	}
}
