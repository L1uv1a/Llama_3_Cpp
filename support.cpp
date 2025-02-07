#include "support.h"
extern double total_matmul_time;
extern unsigned long matmul_call_count;
void softmax(float* x, int size)
{
	// Max value search
	float max_val = x[0];
	for (int i = 1; i < size; i++)
	{
		if (x[i] > max_val) max_val = x[i];
	}
	//Exponential
	float sum = 0.0f;
	for (int i = 0; i < size; i++)
	{
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	for (int i = 0; i < size; i++)
	{
		x[i] /= sum;
	}
}

void matmul(float* xout, float* x, float* weight, int n, int d)
{
	// W (d,n) @ x (n,) -> xout (d,)

	int i;
	matmul_call_count++;
	auto start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) 
	{
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += weight[i * n + j] 
				* x[j];
		}
		xout[i] = val;
	}
	auto end = std::chrono::high_resolution_clock::now();
	total_matmul_time += std::chrono::duration<double>(end - start).count();
}

void matmul_with_debug(float* xout, float* x, float* weight, int n, int d, int layer) {
	if (!xout || !x || !weight) {
		fprintf(stderr, "Invalid pointer passed to matmul at layer %d\n", layer);
		return;
	}

	printf("Running matmul: xout=%p, x=%p, weight=%p\n", xout, x, weight);

	for (int i = 0; i < d; i++) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			// Debugging: Check if indices are within bounds
			if (i * n + j >= d * n) {
				fprintf(stderr, "Access out of bounds: i=%d, j=%d, d=%d, n=%d\n", i, j, d, n);
				return;
			}

			// Debugging: Print the current computation step
			printf("Computing: weight[%d] = %f, x[%d] = %f\n", i * n + j, weight[i * n + j], j, x[j]);

			val += weight[i * n + j] * x[j];
		}
		xout[i] = val;

		// Debugging: Print the result of the current computation
		printf("xout[%d] = %f\n", i, xout[i]);
	}
}


void silu(float* x, int hidden_dim)
{
	for (int i = 0; i < hidden_dim; i++)
	{
		float val = x[i];
		val *= (1.0f / (1.0f + expf(-val)));
	}
}

void elemul(float* x_out, float* x1, float* x2, int size)
{
	for (int i = 0; i < size; i++)
	{
		x_out[i] = x1[i] * x2[i];
	}

}

void rmsnorm(float *o, float *x, float *weight, int size) {
	// calculate sum of squares
	float ss = 0.0f;
	for (int j = 0; j < size; j++) {
		ss += x[j] * x[j];
	}
	ss /= size;
	ss += 1e-5f;
	ss = 1.0f / sqrtf(ss);
	// normalize and scale
	for (int j = 0; j < size; j++) {
		o[j] = weight[j] * (ss * x[j]);
	}
}




