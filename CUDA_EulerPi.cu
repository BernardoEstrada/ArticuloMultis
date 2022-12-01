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

const long MAX_N = 394656595L;
const long N = 394656595L;
const long N2 = N*N;

__global__ void eulerPi(long *sum, long *sums, int n) {
  int i, tid = threadIdx.x + blockIdx.x * blockDim.x;


  for (i = tid; i < n; i += blockDim.x * gridDim.x) {
    if (i != 0)
      sums[tid] += N2 / (i * i);
  }
  __syncthreads();

  if (tid == 0) {
    for (i = 0; i < blockDim.x * gridDim.x; i++) {
      *pi += sums[i];
    }
    *pi *= h;
  }
}

int main(int argc, char* argv[]) {
  long *sum, *sums, *d_sum, *d_sums;
  int i, blockSize, gridSize;
  double pi;

  // allocate memory
  sum = (long *) malloc(sizeof(long));
  sums = (long *) malloc(sizeof(long) * 1024);
  cudaMalloc((void **) &d_sum, sizeof(long));
  cudaMalloc((void **) &d_sums, sizeof(long) * 1024);

  // initialize variables
  *sum = 0;
  for (i = 0; i < 1024; i++) {
    sums[i] = 0;
  }

  // copy data to device
  cudaMemcpy(d_sum, sum, sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sums, sums, sizeof(long) * 1024, cudaMemcpyHostToDevice);

  // calculate pi
  blockSize = 1024;
  gridSize = (N + blockSize - 1) / blockSize;
  eulerPi<<<gridSize, blockSize>>>(d_sum d_sums, N);

  // copy data back to host
  cudaMemcpy(sum, d_sum, sizeof(long), cudaMemcpyDeviceToHost);

  // calculate last step of pi
  pi = sqrt((double)(6 * *sum) / N2);

  // print results
  printf("pi = %lf", *pi);

  // free memory
  free(pi);
  free(sum);
  cudaFree(d_pi);
  cudaFree(d_sum);
  return 0;
}
