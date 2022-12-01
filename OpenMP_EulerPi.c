// =================================================================
//
// File: OpenMP_EulerPi.c
// Author: Bernardo Estrada
// Description: This file contains the code to calculate Pi
//              with Euler's method using OpenMP.
//  To compile:
//      gcc -o OpenMP_EulerPi OpenMP_EulerPi.c -lm -fopenmp
//  To run:
//      ./OpenMP_EulerPi
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_N 394656595L
#define N 394656595L

int main(int argc, char* argv[]) {
  const long N2 = N*N;
  double pi;
  long sum = 0;
  long i;

  if (N > MAX_N) {
    printf("N must be <= %ld (MAX_N)\n", MAX_N);
    exit(1);
  }

  #pragma omp parallel for reduction(+:sum)
  for (i = 0; i < N; i++) {
    if (i == 0) continue;
    sum += N2 / (i * i);
  }
  pi = sqrt((double)(6 * sum)/N2);
  printf("pi = %lf\n", pi);
  return 0;
}