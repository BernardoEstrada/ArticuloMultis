// =================================================================
//
// File: CUDA_EulerPi.c
// Author: Bernardo Estrada
// Description: This file contains the code to calculate Pi
//              with Euler's method using Nvidia's CUDA.
//  To compile:
//      nvcc -o CUDA_EulerPi CUDA_EulerPi.cu
//  To run:
//      ./CUDA_EulerPi
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 1000000000

__global__ void eulerPi(double *pi, double *sum, int n) {
  int i, tid = threadIdx.x + blockIdx.x * blockDim.x;
  double h = 1.0 / n;

  for (i = tid; i < n; i += blockDim.x * gridDim.x) {
    sum[tid] += 4.0 / (1.0 + ((i + 0.5) * h) * ((i + 0.5) * h));
  }
  __syncthreads();

  if (tid == 0) {
    for (i = 0; i < blockDim.x * gridDim.x; i++) {
      *pi += sum[i];
    }
    *pi *= h;
  }
}

int main(int argc, char* argv[]) {
  double *pi, *sum, *d_pi, *d_sum;
  int i, blockSize, gridSize;

  // allocate memory
  pi = (double *) malloc(sizeof(double));
  sum = (double *) malloc(sizeof(double) * 1024);
  cudaMalloc((void **) &d_pi, sizeof(double));
  cudaMalloc((void **) &d_sum, sizeof(double) * 1024);

  // initialize variables
  *pi = 0;
  for (i = 0; i < 1024; i++) {
    sum[i] = 0;
  }

  // copy data to device
  cudaMemcpy(d_pi, pi, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sum, sum, sizeof(double) * 1024, cudaMemcpyHostToDevice);

  // calculate pi
  blockSize = 1024;
  gridSize = (N + blockSize - 1) / blockSize;
  eulerPi<<<gridSize, blockSize>>>(d_pi, d_sum, N);

  // copy data back to host
  cudaMemcpy(pi, d_pi, sizeof(double), cudaMemcpyDeviceToHost);

  // print results
  printf("pi = %lf (error = %lf) \n", *pi, fabs(*pi - M_PI));

  // free memory
  free(pi);
  free(sum);
  cudaFree(d_pi);
  cudaFree(d_sum);
  return 0;
}
