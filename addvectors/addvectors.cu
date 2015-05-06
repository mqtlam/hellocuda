#include <cuda.h>
#include <iostream>

using namespace std;

// simple vector addition kernel
__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
	// compute index from block and thread indices
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundary condition
	if (index < n)
		C[index] = A[index] + B[index];
}

// add to vectors h_A and h_B to get h_C of size n on GPU
void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	const int THREAD_SIZE = 1024;

	// for cudaMalloc and cudaMemcpy
	int size = n * sizeof(float);

	// device variables
	float* d_A;
	float* d_B;
	float* d_C;

	// malloc vector A
	cudaMalloc((void**) &d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	// malloc vector B
	cudaMalloc((void**) &d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// malloc vector C
	cudaMalloc((void**) &d_C, size);

	// run vector addition kernel
	dim3 DimGrid((n-1)/THREAD_SIZE + 1, 1, 1);
	dim3 DimBlock(THREAD_SIZE, 1, 1);
	vecAddKernel<<<DimGrid,DimBlock>>>(d_A, d_B, d_C, n);

	// copy contents
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

// This program adds two vectors together using CUDA.
int main(int argc, char* argv[])
{
	const int VECTOR_SIZE = 10000;

	// create addend vector A
	float* A = new float[VECTOR_SIZE];
	for (int i = 0; i < VECTOR_SIZE; i++)
		A[i] = 2*i;

	// create addend vector B
	float* B = new float[VECTOR_SIZE];
	for (int i = 0; i < VECTOR_SIZE; i++)
		B[i] = 3*i;

	// create sum vector C
	float* C = new float[VECTOR_SIZE];
	for (int i = 0; i < VECTOR_SIZE; i++)
		C[i] = 0;

	// perform vector addition with CUDA
	vecAdd(A, B, C, VECTOR_SIZE);

	// print result
	for (int i = 0; i < VECTOR_SIZE; i++)
		cout << "C[" << i <<  "]=" << C[i] << endl;

	// free memory
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}